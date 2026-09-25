import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_TEST_DIR = tempfile.TemporaryDirectory()
_TEST_DB = Path(_TEST_DIR.name) / "vroomvalue-integration.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"

from apps.api.app.main import app

VEHICLE = {
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


@pytest.fixture(scope="module", autouse=True)
def cleanup_database():
    yield
    _TEST_DIR.cleanup()


def test_happy_path_and_feedback():
    if not Path("models/auto_price.joblib").exists():
        raise AssertionError("Train the model first: python -m ml.training.train")

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200

        before = client.get("/v1/admin/metrics").json()

        response = client.post("/v1/predictions", json=VEHICLE)
        assert response.status_code == 200
        body = response.json()
        assert (
            body["interval_80"]["lower"]
            <= body["estimated_price"]["amount"]
            <= body["interval_80"]["upper"]
        )

        feedback = client.post(
            "/v1/feedback",
            json={
                "prediction_id": body["prediction_id"],
                "actual_sale_price": 28000,
                "sale_date": "2026-09-25",
            },
        )
        assert feedback.status_code == 200

        # Health/admin requests themselves are excluded from service-rate counters.
        client.get("/health/live")
        client.get("/health/ready")
        after = client.get("/v1/admin/metrics").json()

        assert after["prediction_count"] >= 1
        assert after["feedback_count"] >= 1
        assert after["request_count"] == before["request_count"] + 2


def test_engine_size_validation_is_422():
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            json={**VEHICLE, "engine_size": 2.55},
        )
        assert response.status_code == 422
