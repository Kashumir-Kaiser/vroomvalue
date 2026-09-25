from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class VehicleInput(BaseModel):
    make: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=64)
    year: int = Field(ge=2005, le=2024)
    fuel_type: str = Field(min_length=1, max_length=32)
    transmission: str | None = None
    engine_size: float | None = Field(default=None, ge=0.0, le=5.7)
    mileage: int = Field(ge=0, le=2_000_000)
    horsepower: float | None = Field(default=None, gt=0, le=2_000)
    torque: float | None = Field(default=None, gt=0, le=3_000)
    owners: int = Field(ge=1, le=20)
    accident_history: Literal[0, 1] | None = None
    service_history: str | None = None
    color: str | None = None
    body_type: str = Field(min_length=1, max_length=32)
    drivetrain: str = Field(min_length=1, max_length=16)
    fuel_efficiency: float | None = Field(default=None, gt=0, le=500)
    location: str | None = Field(default=None, min_length=2, max_length=32)

    @field_validator("engine_size")
    @classmethod
    def engine_size_one_decimal(cls, value: float | None) -> float | None:
        if value is not None and abs(value * 10 - round(value * 10)) > 1e-9:
            raise ValueError("Engine size must use at most one decimal place (0.1 increments).")
        return value

    @model_validator(mode="after")
    def normalize_strings(self) -> "VehicleInput":
        for field in (
            "make", "model", "fuel_type", "transmission", "service_history", "color",
            "body_type", "drivetrain", "location",
        ):
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, value.strip())
        return self


class FeedbackInput(BaseModel):
    prediction_id: str = Field(min_length=8, max_length=64)
    actual_sale_price: float = Field(gt=0, le=10_000_000)
    sale_date: date


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