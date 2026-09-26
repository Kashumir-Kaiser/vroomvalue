from __future__ import annotations

import os
from datetime import UTC, date, datetime
from functools import lru_cache

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    estimated_price: Mapped[float] = mapped_column(Float)
    interval_lower: Mapped[float] = mapped_column(Float)
    interval_upper: Mapped[float] = mapped_column(Float)
    support: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(32))


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    actual_sale_price: Mapped[float] = mapped_column(Float)
    sale_date: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class FeedbackReviewRecord(Base):
    __tablename__ = "feedback_reviews"

    feedback_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


# 400 is intentionally treated as invalid input for malformed client requests at
# our HTTP boundary (for example, an invalid Content-Length header).
INVALID_INPUT_STATUS_CODES = frozenset({400, 413, 422})


def is_invalid_input_status(status_code: int) -> bool:
    """Classify malformed, oversized, or schema-invalid client requests."""
    return status_code in INVALID_INPUT_STATUS_CODES


class ServiceMetricRecord(Base):
    __tablename__ = "service_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    invalid_input_count: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://vroomvalue:vroomvalue@db:5432/vroomvalue",
    )


@lru_cache(maxsize=4)
def _engine_for_url(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def engine() -> Engine:
    return _engine_for_url(database_url())


def init_db() -> None:
    Base.metadata.create_all(engine())


def database_ready() -> bool:
    try:
        with engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def save_prediction(record: PredictionRecord) -> None:
    with Session(engine()) as session:
        session.add(record)
        session.commit()


def save_feedback(record: FeedbackRecord) -> None:
    with Session(engine()) as session:
        prediction = session.get(PredictionRecord, record.prediction_id)
        if prediction is None:
            raise LookupError("Prediction id was not found.")

        session.add(record)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("Feedback already exists for this prediction id.") from exc


def list_feedback_for_admin() -> list[dict[str, object]]:
    with Session(engine()) as session:
        rows = session.execute(
            select(FeedbackRecord, PredictionRecord, FeedbackReviewRecord)
            .join(
                PredictionRecord,
                FeedbackRecord.prediction_id == PredictionRecord.id,
            )
            .outerjoin(
                FeedbackReviewRecord,
                FeedbackReviewRecord.feedback_id == FeedbackRecord.id,
            )
            .order_by(FeedbackRecord.created_at.desc())
        ).all()

    return [
        {
            "id": feedback.id,
            "prediction_id": feedback.prediction_id,
            "actual_sale_price": feedback.actual_sale_price,
            "sale_date": feedback.sale_date,
            "submitted_at": feedback.created_at,
            "predicted_price": prediction.estimated_price,
            "interval_lower": prediction.interval_lower,
            "interval_upper": prediction.interval_upper,
            "support": prediction.support,
            "model_version": prediction.model_version,
            "review_status": review.status if review is not None else "pending",
            "reviewed_at": review.reviewed_at if review is not None else None,
        }
        for feedback, prediction, review in rows
    ]


def set_feedback_review_status(feedback_id: int, status: str) -> dict[str, object]:
    with Session(engine()) as session:
        feedback = session.get(FeedbackRecord, feedback_id)
        if feedback is None:
            raise LookupError("Feedback id was not found.")

        review = session.get(FeedbackReviewRecord, feedback_id)
        now = datetime.now(UTC)
        if review is None:
            review = FeedbackReviewRecord(
                feedback_id=feedback_id,
                status=status,
                reviewed_at=now,
            )
            session.add(review)
        else:
            review.status = status
            review.reviewed_at = now

        session.commit()
        return {
            "feedback_id": feedback_id,
            "status": review.status,
            "reviewed_at": review.reviewed_at,
        }


def _metric_update_values(
    status_code: int,
    latency_ms: float,
) -> dict[str, object]:
    return {
        "request_count": ServiceMetricRecord.request_count + 1,
        "error_count": ServiceMetricRecord.error_count + int(status_code >= 400),
        "invalid_input_count": (
            ServiceMetricRecord.invalid_input_count
            + int(is_invalid_input_status(status_code))
        ),
        "total_latency_ms": ServiceMetricRecord.total_latency_ms + float(latency_ms),
    }


def record_request_metric(status_code: int, latency_ms: float) -> None:
    """Atomically update process-independent request counters in the database."""
    values = _metric_update_values(status_code, latency_ms)

    with Session(engine()) as session:
        result = session.execute(
            update(ServiceMetricRecord)
            .where(ServiceMetricRecord.id == 1)
            .values(**values)
        )
        if result.rowcount:
            session.commit()
            return

        session.add(
            ServiceMetricRecord(
                id=1,
                request_count=1,
                error_count=int(status_code >= 400),
                invalid_input_count=int(is_invalid_input_status(status_code)),
                total_latency_ms=float(latency_ms),
            )
        )
        try:
            session.commit()
            return
        except IntegrityError:
            # Another worker can create the singleton row between UPDATE and INSERT.
            session.rollback()

        session.execute(
            update(ServiceMetricRecord)
            .where(ServiceMetricRecord.id == 1)
            .values(**values)
        )
        session.commit()


def admin_metrics() -> dict[str, int | float]:
    with Session(engine()) as session:
        predictions = session.scalar(select(func.count()).select_from(PredictionRecord)) or 0
        feedback = session.scalar(select(func.count()).select_from(FeedbackRecord)) or 0
        service = session.get(ServiceMetricRecord, 1)

    requests = int(service.request_count) if service else 0
    errors = int(service.error_count) if service else 0
    invalid = int(service.invalid_input_count) if service else 0
    total_latency = float(service.total_latency_ms) if service else 0.0

    return {
        "prediction_count": int(predictions),
        "feedback_count": int(feedback),
        "request_count": requests,
        "error_rate": round(errors / requests, 4) if requests else 0.0,
        "invalid_input_rate": round(invalid / requests, 4) if requests else 0.0,
        "avg_latency_ms": round(total_latency / requests, 2) if requests else 0.0,
    }
