from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.app.database import (
    FeedbackRecord,
    PredictionRecord,
    admin_metrics,
    init_db,
    save_feedback,
    save_prediction,
)
from apps.api.app.model_runtime import runtime
from apps.api.app.schemas import FeedbackInput, PredictionResponse, VehicleInput

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vroomvalue.api")

service_metrics = {
    "request_count": 0,
    "error_count": 0,
    "invalid_input_count": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.load()
    try:
        init_db()
    except Exception as exc:
        logger.warning(json.dumps({"event": "database.init_failed", "error": str(exc)}))
    yield


app = FastAPI(title="VroomValue API", version="1.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-request-id"],
)


@app.middleware("http")
async def structured_request_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    service_metrics["request_count"] += 1
    service_metrics["total_latency_ms"] += latency_ms
    if response.status_code >= 400:
        service_metrics["error_count"] += 1
    if response.status_code == 422:
        service_metrics["invalid_input_count"] += 1
    response.headers["x-request-id"] = request_id
    logger.info(json.dumps({
        "event": "request.completed",
        "request_id": request_id,
        "route": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
    }))
    return response


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    runtime.load()
    error = runtime.ready_error()
    if error:
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": error})
    return {"status": "ready", "model_version": runtime.bundle["model_version"]}


@app.get("/v1/metadata")
def metadata():
    if runtime.ready_error():
        raise HTTPException(status_code=503, detail=runtime.ready_error())
    return runtime.metadata()


@app.post("/v1/predictions", response_model=PredictionResponse)
def predict(payload: VehicleInput, request: Request):
    if runtime.ready_error():
        raise HTTPException(status_code=503, detail=runtime.ready_error())
    result = runtime.predict(payload)
    prediction_id = uuid.uuid4().hex
    request_id = request.state.request_id
    save_prediction(PredictionRecord(
        id=prediction_id,
        estimated_price=result["estimate"],
        interval_lower=result["lower"],
        interval_upper=result["upper"],
        support=result["support"],
        model_version=runtime.bundle["model_version"],
    ))
    logger.info(json.dumps({
        "event": "prediction.completed",
        "request_id": request_id,
        "model_version": runtime.bundle["model_version"],
        "schema_version": runtime.bundle["schema_version"],
        "support": result["support"],
        "warning_count": len(result["warnings"]),
    }))
    return {
        "prediction_id": prediction_id,
        "request_id": request_id,
        "estimated_price": {"amount": result["estimate"], "currency": "USD"},
        "interval_80": {"lower": result["lower"], "upper": result["upper"]},
        "support": result["support"],
        "top_factors": result["top_factors"],
        "warnings": result["warnings"],
        "model": {
            "name": runtime.bundle["model_name"],
            "version": runtime.bundle["model_version"],
            "schema_version": runtime.bundle["schema_version"],
            "as_of_date": runtime.bundle["as_of_date"],
        },
    }


@app.post("/v1/feedback")
def feedback(payload: FeedbackInput):
    try:
        save_feedback(FeedbackRecord(
            prediction_id=payload.prediction_id,
            actual_sale_price=payload.actual_sale_price,
            sale_date=payload.sale_date,
        ))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "accepted"}


@app.get("/v1/admin/metrics")
def metrics():
    data = admin_metrics()
    requests = service_metrics["request_count"]
    return {
        **data,
        "request_count": requests,
        "error_rate": round(service_metrics["error_count"] / requests, 4) if requests else 0.0,
        "invalid_input_rate": round(service_metrics["invalid_input_count"] / requests, 4) if requests else 0.0,
        "avg_latency_ms": round(service_metrics["total_latency_ms"] / requests, 2) if requests else 0.0,
        "current_model_version": runtime.bundle["model_version"] if runtime.bundle else None,
        "readiness": "ready" if not runtime.ready_error() else "not_ready",
    }