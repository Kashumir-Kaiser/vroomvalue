import math

import pytest
from pydantic import ValidationError

from apps.api.app.schemas import FeedbackInput, FeedbackReviewInput, VehicleInput

BASE = {
    "make": "Toyota",
    "model": "Camry",
    "year": 2022,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "engine_size": 2.5,
    "mileage": 42000,
    "horsepower": 203.0,
    "torque": 184.0,
    "owners": 1,
    "accident_history": 0,
    "service_history": "Full Service",
    "color": "White",
    "body_type": "Sedan",
    "drivetrain": "FWD",
    "fuel_efficiency": 32.0,
    "location": "TX",
}


def test_engine_size_tenth_is_valid():
    assert VehicleInput(**BASE).engine_size == 2.5


def test_engine_size_two_decimals_is_rejected():
    with pytest.raises(ValidationError, match="one decimal place"):
        VehicleInput(**{**BASE, "engine_size": 2.55})


@pytest.mark.parametrize(
    "field",
    ["engine_size", "horsepower", "torque", "fuel_efficiency"],
)
def test_non_finite_numbers_are_rejected(field: str):
    with pytest.raises(ValidationError):
        VehicleInput(**{**BASE, field: math.inf})


def test_whitespace_required_string_is_rejected_with_specific_message():
    with pytest.raises(ValidationError, match="Field must not be blank"):
        VehicleInput(**{**BASE, "make": "   "})


def test_optional_blank_becomes_unknown_and_location_is_canonicalized():
    payload = VehicleInput(**{**BASE, "color": "  ", "location": " tx "})
    assert payload.color is None
    assert payload.location == "TX"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        VehicleInput(**{**BASE, "unexpected": "value"})


def test_future_feedback_date_is_rejected():
    with pytest.raises(ValidationError, match="future"):
        FeedbackInput(
            prediction_id="abc12345",
            actual_sale_price=10_000,
            sale_date="2999-01-01",
        )



@pytest.mark.parametrize("price", [0, -0.01, -100])
def test_non_positive_feedback_price_is_rejected(price: float):
    with pytest.raises(ValidationError):
        FeedbackInput(
            prediction_id="abc12345",
            actual_sale_price=price,
            sale_date="2026-09-25",
        )


def test_feedback_date_before_automobile_era_is_rejected():
    with pytest.raises(ValidationError, match="1886-01-29"):
        FeedbackInput(
            prediction_id="abc12345",
            actual_sale_price=10_000,
            sale_date="1800-01-01",
        )


def test_earliest_supported_feedback_date_is_valid():
    payload = FeedbackInput(
        prediction_id="abc12345",
        actual_sale_price=10_000,
        sale_date="1886-01-29",
    )
    assert payload.sale_date.isoformat() == "1886-01-29"



def test_feedback_review_status_is_limited_to_supported_actions():
    assert FeedbackReviewInput(status="accepted").status == "accepted"
    assert FeedbackReviewInput(status="closed").status == "closed"
    with pytest.raises(ValidationError):
        FeedbackReviewInput(status="pending")
