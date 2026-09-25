from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VehicleInput(StrictInputModel):
    make: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=64)
    year: int = Field(ge=2005, le=2024)
    fuel_type: str = Field(min_length=1, max_length=32)
    transmission: str | None = Field(default=None, max_length=32)
    engine_size: float | None = Field(default=None, ge=0.0, le=5.7)
    mileage: int = Field(ge=0, le=2_000_000)
    horsepower: float | None = Field(default=None, gt=0, le=2_000)
    torque: float | None = Field(default=None, gt=0, le=3_000)
    owners: int = Field(ge=1, le=5)
    accident_history: Literal[0, 1] | None = None
    service_history: str | None = Field(default=None, max_length=64)
    color: str | None = Field(default=None, max_length=64)
    body_type: str = Field(min_length=1, max_length=32)
    drivetrain: str = Field(min_length=1, max_length=16)
    fuel_efficiency: float | None = Field(default=None, gt=0, le=500)
    location: str | None = Field(default=None, max_length=32)

    @field_validator(
        "make",
        "model",
        "fuel_type",
        "transmission",
        "service_history",
        "color",
        "body_type",
        "drivetrain",
        "location",
        mode="before",
    )
    @classmethod
    def normalize_text_input(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("make", "model", "fuel_type", "body_type", "drivetrain")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("Field must not be blank.")
        return value

    @field_validator("location")
    @classmethod
    def normalize_location(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None

    @field_validator("engine_size")
    @classmethod
    def engine_size_one_decimal(cls, value: float | None) -> float | None:
        if value is not None and abs(value * 10 - round(value * 10)) > 1e-9:
            raise ValueError("Engine size must use at most one decimal place (0.1 increments).")
        return value

    @model_validator(mode="after")
    def electric_engine_consistency(self) -> VehicleInput:
        # Zero is valid for electric vehicles. Contradictory non-electric zero is
        # deliberately not rejected: serving marks it low-support as specified.
        return self


class FeedbackInput(StrictInputModel):
    prediction_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    actual_sale_price: float = Field(gt=0, le=10_000_000)
    sale_date: date

    @field_validator("sale_date")
    @classmethod
    def sale_date_not_in_future(cls, value: date) -> date:
        if value > datetime.now(UTC).date():
            raise ValueError("Sale date cannot be in the future.")
        return value


class PriceAmount(BaseModel):
    amount: float
    currency: Literal["USD"] = "USD"


class Interval80(BaseModel):
    lower: float
    upper: float


class Factor(BaseModel):
    feature: str
    direction: Literal["up", "down"]


class PredictionResponse(BaseModel):
    prediction_id: str
    request_id: str
    estimated_price: PriceAmount
    interval_80: Interval80
    support: Literal["in_distribution", "low_confidence"]
    top_factors: list[Factor]
    warnings: list[str]
    model: dict[str, str]


class AdminMetricsResponse(BaseModel):
    prediction_count: int
    feedback_count: int
    request_count: int
    error_rate: float
    invalid_input_rate: float
    avg_latency_ms: float
    current_model_version: str | None
    readiness: Literal["ready", "not_ready"]
