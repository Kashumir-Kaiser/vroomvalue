import numpy as np
import pandas as pd
import pytest

import apps.api.app.model_runtime as model_runtime
from apps.api.app.model_runtime import Runtime
from apps.api.app.schemas import VehicleInput


class FakeModel:
    def predict(self, features):
        return np.array([10_000.0])


class ExplodingExplainer:
    def shap_values(self, features):
        raise RuntimeError("simulated SHAP failure")


def _runtime_with_bundle(feature_columns: list[str]) -> Runtime:
    runtime = Runtime()
    runtime.bundle = {
        "model": FakeModel(),
        "objective": "raw",
        "reference_year": 2026,
        "feature_columns": feature_columns,
        "interval": {
            "edges": [-np.inf, np.inf],
            "quantiles": [500.0],
        },
    }
    return runtime


def test_feature_alignment_reorders_to_saved_schema():
    runtime = _runtime_with_bundle(["b", "a"])
    features = pd.DataFrame([{"a": 1, "b": 2}])

    aligned = runtime._align_features(features)

    assert list(aligned.columns) == ["b", "a"]


def test_feature_alignment_rejects_extra_or_missing_features():
    runtime = _runtime_with_bundle(["a", "b"])

    with pytest.raises(RuntimeError, match="feature schema mismatch"):
        runtime._align_features(pd.DataFrame([{"a": 1, "c": 2}]))


def test_shap_failure_does_not_fail_prediction(monkeypatch):
    runtime = _runtime_with_bundle([])
    runtime.bundle["feature_columns"] = ["only_feature"]
    runtime._explainer = ExplodingExplainer()
    monkeypatch.setattr(
        "apps.api.app.model_runtime.build_features",
        lambda frame, reference_year: pd.DataFrame([{"only_feature": 1.0}]),
    )
    monkeypatch.setattr(
        runtime,
        "_support",
        lambda payload: ("in_distribution", []),
    )

    payload = VehicleInput(
        make="Toyota",
        model="Camry",
        year=2022,
        fuel_type="Petrol",
        transmission="Auto",
        engine_size=2.5,
        mileage=42000,
        horsepower=203.0,
        torque=184.0,
        owners=1,
        accident_history=0,
        service_history="Full Service",
        color="White",
        body_type="Sedan",
        drivetrain="FWD",
        fuel_efficiency=32.0,
        location="TX",
    )

    result = runtime.predict(payload)

    assert result["estimate"] == 10_000.0
    assert result["top_factors"] == []
    assert "Explanation unavailable for this prediction." in result["warnings"]


def test_lowercase_electric_does_not_add_non_electric_zero_warning():
    runtime = Runtime()
    runtime.bundle = {
        "support": {
            "categories": {
                "Make": ["Tesla"],
                "Model": ["Model 3"],
                "Fuel_Type": ["Electric"],
                "Transmission": [],
                "Service_History": [],
                "Color": [],
                "Body_Type": ["Sedan"],
                "Drivetrain": ["RWD"],
                "Location": [],
            },
            "make_models": {"Tesla": ["Model 3"]},
            "numeric": {
                "Year": {"min": 2020, "max": 2024},
                "Engine_Size": {"min": 0.0, "max": 5.7},
                "Mileage": {"min": 0, "max": 100000},
                "Horsepower": {"min": 100, "max": 500},
                "Torque": {"min": 100, "max": 500},
                "Owners": {"min": 1, "max": 5},
                "Fuel_Efficiency": {"min": 10, "max": 150},
            },
        }
    }
    payload = VehicleInput(
        make="Tesla",
        model="Model 3",
        year=2022,
        fuel_type="electric",
        transmission=None,
        engine_size=0.0,
        mileage=10000,
        horsepower=250,
        torque=250,
        owners=1,
        accident_history=0,
        service_history=None,
        color=None,
        body_type="Sedan",
        drivetrain="RWD",
        fuel_efficiency=120,
        location=None,
    )

    _, warnings = runtime._support(payload)

    assert "Engine_Size 0.0 was treated as low-support for a non-electric vehicle." not in warnings



def test_refresh_if_changed_throttles_file_checks_within_ttl(monkeypatch):
    runtime = Runtime(refresh_ttl_seconds=1.0)
    calls = {"dataset": 0, "model": 0}
    times = iter([10.0, 10.25, 11.25])

    monkeypatch.setattr(
        "apps.api.app.model_runtime.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        runtime,
        "_load_dataset_gate",
        lambda force=False: calls.__setitem__("dataset", calls["dataset"] + 1),
    )
    monkeypatch.setattr(
        runtime,
        "_load_model",
        lambda force=False: calls.__setitem__("model", calls["model"] + 1),
    )

    runtime.refresh_if_changed()
    runtime.refresh_if_changed()
    runtime.refresh_if_changed()

    assert calls == {"dataset": 2, "model": 2}


def test_support_uses_explicit_vehicle_model_mapping():
    runtime = Runtime()
    runtime.bundle = {
        "support": {
            "categories": {
                "Make": ["Toyota"],
                "Model": ["Camry"],
                "Fuel_Type": ["Petrol"],
                "Transmission": ["Auto"],
                "Service_History": ["Full Service"],
                "Color": ["White"],
                "Body_Type": ["Sedan"],
                "Drivetrain": ["FWD"],
                "Location": ["TX"],
            },
            "make_models": {"Toyota": ["Camry"]},
            "numeric": {
                "Year": {"min": 2005, "max": 2024},
                "Engine_Size": {"min": 0.0, "max": 5.7},
                "Mileage": {"min": 0, "max": 536731},
                "Horsepower": {"min": 97, "max": 434},
                "Torque": {"min": 77, "max": 443},
                "Owners": {"min": 1, "max": 5},
                "Fuel_Efficiency": {"min": 10, "max": 141},
            },
        }
    }
    payload = VehicleInput(
        make="Toyota",
        model="Corolla",
        year=2022,
        fuel_type="Petrol",
        transmission="Auto",
        engine_size=2.0,
        mileage=10000,
        horsepower=150,
        torque=150,
        owners=1,
        accident_history=0,
        service_history="Full Service",
        color="White",
        body_type="Sedan",
        drivetrain="FWD",
        fuel_efficiency=32,
        location="TX",
    )

    support, warnings = runtime._support(payload)

    assert support == "low_confidence"
    assert "Model was not observed in training data." in warnings
    assert "This Make-Model pairing was not observed in training data." in warnings



def test_refresh_without_prior_load_populates_missing_file_errors(
    monkeypatch,
    tmp_path,
):
    runtime = Runtime(refresh_ttl_seconds=0)
    missing_dataset = tmp_path / "missing.csv"
    missing_model = tmp_path / "missing.joblib"

    monkeypatch.setattr(model_runtime, "DATA_PATH", missing_dataset)
    monkeypatch.setattr(model_runtime, "MODEL_PATH", missing_model)

    runtime.refresh_if_changed()

    assert runtime.dataset_error == (
        f"Dataset not found at {missing_dataset.as_posix()}. Add the file and retry."
    )
    assert runtime.model_error == (
        f"Model artifact not found at {missing_model.as_posix()}. "
        "Train the model and retry."
    )
    assert runtime.ready_error() is not None


@pytest.mark.parametrize("raw", ["not-a-number", "nan", "inf", "-inf"])
def test_invalid_refresh_ttl_falls_back_to_default(monkeypatch, raw: str):
    monkeypatch.setenv("RUNTIME_REFRESH_TTL_SECONDS", raw)

    assert (
        model_runtime._read_refresh_ttl_seconds()
        == model_runtime.DEFAULT_REFRESH_TTL_SECONDS
    )


def test_negative_refresh_ttl_is_clamped_to_zero(monkeypatch):
    monkeypatch.setenv("RUNTIME_REFRESH_TTL_SECONDS", "-2.5")

    assert model_runtime._read_refresh_ttl_seconds() == 0.0


def test_failed_model_load_retries_same_signature_after_ttl(monkeypatch):
    runtime = Runtime(refresh_ttl_seconds=0)
    fixed_signature = (1, 2, 3)
    attempts = {"count": 0}

    monkeypatch.setattr(model_runtime, "_file_signature", lambda path: fixed_signature)
    monkeypatch.setattr(model_runtime, "verify_checksum", lambda path: "ok")
    monkeypatch.setattr(model_runtime.shap, "TreeExplainer", lambda model: object())

    bundle = {
        "model": FakeModel(),
        "objective": "raw",
        "reference_year": 2026,
        "interval": {"edges": [-np.inf, np.inf], "quantiles": [500.0]},
        "support": {},
        "feature_columns": ["feature"],
        "model_name": "auto_price",
        "model_version": "1",
        "schema_version": "1.0",
        "as_of_date": "2026-09-26",
    }

    def flaky_load(path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("transient read failure")
        return bundle

    monkeypatch.setattr(model_runtime.joblib, "load", flaky_load)

    runtime.refresh_if_changed()
    assert runtime.bundle is None
    assert "transient read failure" in (runtime.model_error or "")

    runtime.refresh_if_changed()

    assert attempts["count"] == 2
    assert runtime.bundle is bundle
    assert runtime.model_error is None



def test_runtime_reads_ttl_environment_when_constructed(monkeypatch):
    monkeypatch.setenv("RUNTIME_REFRESH_TTL_SECONDS", "2.5")

    runtime = Runtime()

    assert runtime._refresh_ttl_seconds == 2.5


def test_persistent_model_failure_logs_are_rate_limited(monkeypatch, caplog):
    runtime = Runtime(refresh_ttl_seconds=1.0)
    fixed_signature = (7, 8, 9)
    times = iter([10.0, 20.0, 71.0])

    monkeypatch.setattr(model_runtime, "_file_signature", lambda path: fixed_signature)
    def fail_checksum(path):
        raise ValueError("corrupt artifact")

    monkeypatch.setattr(model_runtime, "verify_checksum", fail_checksum)
    monkeypatch.setattr(
        model_runtime.time,
        "monotonic",
        lambda: next(times),
    )

    with caplog.at_level("ERROR", logger="vroomvalue.runtime"):
        runtime._load_model()
        runtime._load_model()
        runtime._load_model()

    failure_logs = [
        record.message
        for record in caplog.records
        if '"event": "model.load_failed"' in record.message
    ]

    assert len(failure_logs) == 2
    assert '"consecutive_failures": 1' in failure_logs[0]
    assert '"consecutive_failures": 3' in failure_logs[1]


def test_model_recovery_log_resets_failure_counter(monkeypatch, caplog):
    runtime = Runtime(refresh_ttl_seconds=0)
    fixed_signature = (4, 5, 6)
    attempts = {"count": 0}

    monkeypatch.setattr(model_runtime, "_file_signature", lambda path: fixed_signature)
    monkeypatch.setattr(model_runtime.shap, "TreeExplainer", lambda model: object())

    bundle = {
        "model": FakeModel(),
        "objective": "raw",
        "reference_year": 2026,
        "interval": {"edges": [-np.inf, np.inf], "quantiles": [500.0]},
        "support": {},
        "feature_columns": ["feature"],
        "model_name": "auto_price",
        "model_version": "1",
        "schema_version": "1.0",
        "as_of_date": "2026-09-26",
    }

    def flaky_verify(path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("corrupt artifact")
        return "ok"

    monkeypatch.setattr(model_runtime, "verify_checksum", flaky_verify)
    monkeypatch.setattr(model_runtime.joblib, "load", lambda path: bundle)

    with caplog.at_level("INFO", logger="vroomvalue.runtime"):
        runtime._load_model()
        runtime._load_model()

    assert runtime._model_load_failure_count == 0
    assert runtime._last_model_failure_log_at is None
    assert any(
        '"event": "model.load_recovered"' in record.message
        for record in caplog.records
    )
