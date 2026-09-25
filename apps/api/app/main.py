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
from starlette.background import BackgroundTask, BackgroundTasks

from apps.api.app.database import (
    FeedbackRecord,
    PredictionRecord,
    admin_metrics,
    init_db,
    record_request_metric,
    save_feedback,
    save_prediction,
)
from apps.api.app.middleware import BodySizeLimitMiddleware
from apps.api.app.model_runtime import runtime
from apps.api.app.schemas import (
    AdminMetricsResponse,
    FeedbackInput,
    PredictionResponse,
    VehicleInput,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vroomvalue.api")

MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "32768"))
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
METRICS_EXCLUDED_PATHS = {
    "/health/live",
    "/health/ready",
    "/v1/admin/metrics",
}


def _request_id(value: str | None) -> str:
    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid.uuid4().hex


def _require_admin_token(provided: str | None) -> None:
    if ADMIN_TOKEN and (provided is None or not hmac.compare_digest(provided, ADMIN_TOKEN)):
        raise HTTPException(status_code=401, detail="Admin authentication required.")


def _persist_request_metric(path: str, status_code: int, latency_ms: float) -> None:
    if path in METRICS_EXCLUDED_PATHS:
        return
    try:
        record_request_metric(status_code, latency_ms)
    except SQLAlchemyError:
        logger.exception(
            json.dumps(
                {
                    "event": "request.metric_persistence_failed",
                    "route": path,
                    "status_code": status_code,
                }
            )
        )


def _attach_metric_background(
    response,
    path: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Persist metrics after the response body is sent, off the event loop."""
    if path in METRICS_EXCLUDED_PATHS:
        return

    task = BackgroundTask(_persist_request_metric, path, status_code, latency_ms)
    if response.background is None:
        response.background = task
    elif isinstance(response.background, BackgroundTasks):
        response.background.tasks.append(task)
    else:
        response.background = BackgroundTasks([response.background, task])


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.load(force=True)
    try:
        init_db()
    except SQLAlchemyError:
        logger.exception(json.dumps({"event": "database.init_failed"}))
    yield


app = FastAPI(title="VroomValue API", version="1.0.0", lifespan=lifespan)
origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# The body limiter sits inside CORS so even direct 400/413 responses include
# browser-readable CORS headers. Request logging sits between them.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)


@app.middleware("http")
async def request_guard_and_log(request: Request, call_next):
    request_id = _request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            json.dumps(
                {
                    "event": "request.failed",
                    "request_id": request_id,
                    "route": request.url.path,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={"x-request-id": request_id},
        )
        _attach_metric_background(
            response,
            request.url.path,
            500,
            latency_ms,
        )
        return response

    latency_ms = (time.perf_counter() - started) * 1000
    _attach_metric_background(
        response,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    response.headers["x-request-id"] = request_id
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["referrer-policy"] = "no-referrer"
    response.headers["x-frame-options"] = "DENY"
    logger.info(
        json.dumps(
            {
                "event": "request.completed",
                "request_id": request_id,
                "route": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            }
        )
    )
    return response


# Added last so it is the outermost user middleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-request-id", "x-admin-token"],
)


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
        save_prediction(
            PredictionRecord(
                id=prediction_id,
                estimated_price=result["estimate"],
                interval_lower=result["lower"],
                interval_upper=result["upper"],
                support=result["support"],
                model_version=runtime.bundle["model_version"],
            )
        )
    except SQLAlchemyError as exc:
        logger.error(
            json.dumps(
                {
                    "event": "prediction.persistence_failed",
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                }
            )
        )
        raise HTTPException(
            status_code=503,
            detail="Prediction storage is unavailable.",
        ) from exc

    logger.info(
        json.dumps(
            {
                "event": "prediction.completed",
                "request_id": request_id,
                "model_version": runtime.bundle["model_version"],
                "schema_version": runtime.bundle["schema_version"],
                "support": result["support"],
                "warning_count": len(result["warnings"]),
            }
        )
    )
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
        save_feedback(
            FeedbackRecord(
                prediction_id=payload.prediction_id,
                actual_sale_price=payload.actual_sale_price,
                sale_date=payload.sale_date,
            )
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage is unavailable.",
        ) from exc
    return {"status": "accepted"}


@app.get("/v1/admin/metrics", response_model=AdminMetricsResponse)
def metrics(x_admin_token: str | None = Header(default=None)) -> AdminMetricsResponse:
    _require_admin_token(x_admin_token)
    try:
        data = admin_metrics()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Metrics storage is unavailable.",
        ) from exc

    return AdminMetricsResponse(
        **data,
        current_model_version=(
            runtime.bundle["model_version"] if runtime.bundle else None
        ),
        readiness="ready" if not runtime.ready_error() else "not_ready",
    )
