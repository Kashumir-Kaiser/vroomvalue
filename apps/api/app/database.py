from __future__ import annotations

import os
from datetime import UTC, date, datetime
from functools import lru_cache

from sqlalchemy import Date, DateTime, Float, Integer, String, create_engine, func, select, text
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


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://vroomvalue:vroomvalue@db:5432/vroomvalue",
    )


@lru_cache(maxsize=4)
def _engine_for_url(url: str) -> Engine:
    # Reuse the SQLAlchemy pool instead of constructing a new engine for every request.
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


def admin_metrics() -> dict[str, int]:
    with Session(engine()) as session:
        predictions = session.scalar(select(func.count()).select_from(PredictionRecord)) or 0
        feedback = session.scalar(select(func.count()).select_from(FeedbackRecord)) or 0
    return {"prediction_count": int(predictions), "feedback_count": int(feedback)}
