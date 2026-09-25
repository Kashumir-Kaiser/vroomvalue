from __future__ import annotations

import hmac
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

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

MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "32768"))
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")

service_metrics = {
    "request_count": 0,
    "error_count": 0,
    "invalid_input_count": 0,
    "total_latency_ms": 0.0,
}


def _request_id(value: str | None) -> str:
    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid.uuid4().hex


def _require_admin_token(provided: str | None) -> None:
    # Local portfolio mode remains usable with no token configured. Any deployed
    # environment can make this endpoint private by setting ADMIN_TOKEN.
    if ADMIN_TOKEN and (provided is None or not hmac.compare_digest(provided, ADMIN_TOKEN)):
        raise HTTPException(status_code=401, detail="Admin authentication required.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.load(force=True)
    try:
        init_db()
    except Exception:
        logger.exception(json.dumps({"event": "database.init_failed"}))
    yield


app = FastAPI(title="VroomValue API", version="1.0.0", lifespan=lifespan)
origins = [
    x.strip()
    for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-request-id", "x-admin-token"],
)


@app.middleware("http")
async def request_guard_and_log(request: Request, call_next):
    request_id = _request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes."},
                    headers={"x-request-id": request_id},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header."},
                headers={"x-request-id": request_id},
            )

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        service_metrics["request_count"] += 1
        service_metrics["error_count"] += 1
        service_metrics["total_latency_ms"] += latency_ms
        logger.exception(json.dumps({
            "event": "request.failed",
            "request_id": request_id,
            "route": request.url.path,
            "latency_ms": round(latency_ms, 2),
        }))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    latency_ms = (time.perf_counter() - started) * 1000
    service_metrics["request_count"] += 1
    service_metrics["total_latency_ms"] += latency_ms
    if response.status_code >= 400:
        service_metrics["error_count"] += 1
    if response.status_code == 422:
        service_metrics["invalid_input_count"] += 1

    response.headers["x-request-id"] = request_id
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["referrer-policy"] = "no-referrer"
    response.headers["x-frame-options"] = "DENY"
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
    runtime.refresh_if_changed()
    error = runtime.ready_error()
    if error:
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": error})
    assert runtime.bundle is not None
    return {"status": "ready", "model_version": runtime.bundle["model_version"]}


@app.get("/v1/metadata")
def metadata():
    runtime.refresh_if_changed()
    error = runtime.ready_error()
    if error:
        raise HTTPException(status_code=503, detail=error)
    return runtime.metadata()


@app.post("/v1/predictions", response_model=PredictionResponse)
def predict(payload: VehicleInput, request: Request):
    runtime.refresh_if_changed()
    error = runtime.ready_error()
    if error:
        raise HTTPException(status_code=503, detail=error)

    result = runtime.predict(payload)
    prediction_id = uuid.uuid4().hex
    request_id = request.state.request_id
    assert runtime.bundle is not None

    try:
        save_prediction(PredictionRecord(
            id=prediction_id,
            estimated_price=result["estimate"],
            interval_lower=result["lower"],
            interval_upper=result["upper"],
            support=result["support"],
            model_version=runtime.bundle["model_version"],
        ))
    except SQLAlchemyError as exc:
        logger.error(json.dumps({
            "event": "prediction.persistence_failed",
            "request_id": request_id,
            "error_type": type(exc).__name__,
        }))
        raise HTTPException(status_code=503, detail="Prediction storage is unavailable.") from exc

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
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Feedback storage is unavailable.") from exc
    return {"status": "accepted"}


@app.get("/v1/admin/metrics")
def metrics(x_admin_token: str | None = Header(default=None)):
    _require_admin_token(x_admin_token)
    try:
        data = admin_metrics()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Metrics storage is unavailable.") from exc

    requests = service_metrics["request_count"]
    return {
        **data,
        "request_count": requests,
        "error_rate": round(service_metrics["error_count"] / requests, 4) if requests else 0.0,
        "invalid_input_rate": (
            round(service_metrics["invalid_input_count"] / requests, 4) if requests else 0.0
        ),
        "avg_latency_ms": (
            round(service_metrics["total_latency_ms"] / requests, 2) if requests else 0.0
        ),
        "current_model_version": runtime.bundle["model_version"] if runtime.bundle else None,
        "readiness": "ready" if not runtime.ready_error() else "not_ready",
    }
