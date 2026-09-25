import numpy as np
import pandas as pd
import pytest

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
