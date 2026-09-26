import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_TEST_DIR = tempfile.TemporaryDirectory()
_TEST_DB = Path(_TEST_DIR.name) / "vroomvalue-integration.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"

from apps.api.app.main import MAX_BODY_BYTES, app

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
        # Metric persistence is asynchronous so response completion is never held
        # open by a database write.
        client.get("/health/live")
        client.get("/health/ready")
        deadline = time.monotonic() + 2.0
        while True:
            after = client.get("/v1/admin/metrics").json()
            if after["request_count"] >= before["request_count"] + 2:
                break
            if time.monotonic() >= deadline:
                raise AssertionError("request metrics were not persisted in time")
            time.sleep(0.02)

        assert after["prediction_count"] >= 1
        assert after["feedback_count"] >= 1
        assert after["request_count"] == before["request_count"] + 2

        review_queue = client.get("/v1/admin/feedback")
        assert review_queue.status_code == 200
        review_item = next(
            item
            for item in review_queue.json()
            if item["prediction_id"] == body["prediction_id"]
        )
        assert review_item["review_status"] == "pending"

        accepted = client.post(
            f"/v1/admin/feedback/{review_item['id']}/review",
            json={"status": "accepted"},
        )
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "accepted"

        refreshed_queue = client.get("/v1/admin/feedback").json()
        refreshed_item = next(
            item
            for item in refreshed_queue
            if item["prediction_id"] == body["prediction_id"]
        )
        assert refreshed_item["review_status"] == "accepted"


def test_engine_size_validation_is_422():
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            json={**VEHICLE, "engine_size": 2.55},
        )
        assert response.status_code == 422



def test_oversize_response_preserves_cors_headers():
    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            content=b"x" * (MAX_BODY_BYTES + 1),
            headers={
                "content-type": "application/json",
                "origin": "http://localhost:3000",
            },
        )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.json()["detail"] == f"Request body exceeds {MAX_BODY_BYTES} bytes."


def test_admin_metrics_openapi_schema_is_explicit():
    schema = app.openapi()
    response_schema = schema["paths"]["/v1/admin/metrics"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert response_schema["$ref"].endswith("/AdminMetricsResponse")



def test_readiness_includes_database_health(monkeypatch):
    monkeypatch.setattr("apps.api.app.main.database_ready", lambda: False)

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "Database is unavailable.",
    }



def test_health_live_exposes_handler_server_timing():
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    server_timing = response.headers["server-timing"]
    assert "dispatch-wait;dur=" in server_timing
    assert "handler-total;dur=" in server_timing



def test_metadata_exposes_full_vehicle_choices():
    with TestClient(app) as client:
        response = client.get("/v1/metadata")

    assert response.status_code == 200
    metadata = response.json()
    makes = metadata["categories"]["Make"]
    assert len(makes) > 1
    assert "Toyota" in makes
    assert set(metadata["make_models"]["Toyota"]) >= {
        "Camry",
        "Corolla",
        "Highlander",
        "RAV4",
    }
