from __future__ import annotations

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from apps.api.app.database import (
    FeedbackRecord,
    PredictionRecord,
    admin_metrics,
    database_ready,
    init_db,
    list_feedback_for_admin,
    record_request_metric,
    save_feedback,
    save_prediction,
    set_feedback_review_status,
)
from apps.api.app.middleware import (
    BodySizeLimitMiddleware,
    RequestObservabilityMiddleware,
)
from apps.api.app.model_runtime import runtime
from apps.api.app.schemas import (
    AdminFeedbackItem,
    AdminMetricsResponse,
    FeedbackInput,
    FeedbackReviewInput,
    PredictionResponse,
    VehicleInput,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vroomvalue.api")

DEFAULT_MAX_BODY_BYTES = 32768


def _read_max_body_bytes() -> int:
    raw = os.getenv("MAX_BODY_BYTES")
    if raw is None:
        return DEFAULT_MAX_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MAX_BODY_BYTES=%r; using %d.",
            raw,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    if value <= 0:
        logger.warning(
            "Non-positive MAX_BODY_BYTES=%r; using %d.",
            raw,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    return value


MAX_BODY_BYTES = _read_max_body_bytes()
REQUEST_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
METRICS_EXCLUDED_PATHS = {
    "/health/live",
    "/health/ready",
    "/v1/admin/metrics",
    "/v1/admin/feedback",
}


def _require_admin_token(provided: str | None) -> None:
    if ADMIN_TOKEN and (provided is None or not hmac.compare_digest(provided, ADMIN_TOKEN)):
        raise HTTPException(status_code=401, detail="Admin authentication required.")


def _persist_request_metric(_path: str, status_code: int, latency_ms: float) -> None:
    # The path argument is retained for the middleware MetricRecorder contract.
    # Path exclusion is owned by RequestObservabilityMiddleware.
    record_request_metric(status_code, latency_ms)


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

# Middleware order is deliberate:
# CORS (outermost) -> observability -> body limit -> FastAPI routes.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
app.add_middleware(
    RequestObservabilityMiddleware,
    metric_recorder=_persist_request_metric,
    excluded_paths=METRICS_EXCLUDED_PATHS,
    request_id_pattern=REQUEST_ID_PATTERN,
)
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
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": error},
        )
    if not database_ready():
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Database is unavailable."},
        )
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
    prediction_id = os.urandom(16).hex()
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



@app.get("/v1/admin/feedback", response_model=list[AdminFeedbackItem])
def admin_feedback(
    x_admin_token: str | None = Header(default=None),
) -> list[AdminFeedbackItem]:
    _require_admin_token(x_admin_token)
    try:
        rows = list_feedback_for_admin()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Feedback review queue is unavailable.",
        ) from exc
    return [AdminFeedbackItem(**row) for row in rows]


@app.post("/v1/admin/feedback/{feedback_id}/review")
def review_feedback(
    feedback_id: int,
    payload: FeedbackReviewInput,
    x_admin_token: str | None = Header(default=None),
):
    _require_admin_token(x_admin_token)
    try:
        return set_feedback_review_status(feedback_id, payload.status)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Feedback review update is unavailable.",
        ) from exc
