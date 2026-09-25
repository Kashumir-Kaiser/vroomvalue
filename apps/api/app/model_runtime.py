from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from apps.api.app.schemas import VehicleInput
from ml.artifacts import verify_checksum
from ml.contracts.schema import DATASET_PATH, DatasetContractError, validate_dataset_contract
from ml.features.build import build_features

MODEL_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", "models/auto_price.joblib"))
DATA_PATH = Path(os.getenv("DATASET_PATH", str(DATASET_PATH)))


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


class Runtime:
    def __init__(self) -> None:
        self.bundle: dict[str, Any] | None = None
        self.dataset_error: str | None = None
        self.model_error: str | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._dataset_signature: tuple[int, int, int] | None = None
        self._model_signature: tuple[int, int, int] | None = None
        self._lock = threading.RLock()

    def _load_dataset_gate(self, *, force: bool = False) -> None:
        signature = _file_signature(DATA_PATH)
        if not force and signature == self._dataset_signature:
            return
        try:
            validate_dataset_contract(DATA_PATH)
            self.dataset_error = None
        except DatasetContractError as exc:
            self.dataset_error = str(exc)
        self._dataset_signature = signature

    def _load_model(self, *, force: bool = False) -> None:
        signature = _file_signature(MODEL_PATH)
        if not force and signature == self._model_signature:
            return

        if signature is None:
            self.model_error = (
                f"Model artifact not found at {MODEL_PATH.as_posix()}. Train the model and retry."
            )
            self.bundle = None
            self._explainer = None
            self._model_signature = None
            return

        try:
            verify_checksum(MODEL_PATH)
            bundle = joblib.load(MODEL_PATH)
            required = {
                "model", "objective", "reference_year", "interval", "support",
                "model_name", "model_version", "schema_version", "as_of_date",
            }
            missing = sorted(required.difference(bundle))
            if missing:
                raise ValueError(f"artifact missing keys: {', '.join(missing)}")
            self.bundle = bundle
            self._explainer = shap.TreeExplainer(bundle["model"])
            self.model_error = None
        except Exception as exc:  # noqa: BLE001 - fail closed on any artifact-load failure
            self.bundle = None
            self._explainer = None
            self.model_error = f"Model artifact could not be loaded: {exc}"
        self._model_signature = signature

    def load(self, *, force: bool = False) -> None:
        with self._lock:
            self._load_dataset_gate(force=force)
            self._load_model(force=force)

    def refresh_if_changed(self) -> None:
        self.load(force=False)

    def ready_error(self) -> str | None:
        return self.dataset_error or self.model_error

    def metadata(self) -> dict[str, Any]:
        if not self.bundle:
            raise RuntimeError(self.ready_error() or "Model unavailable")
        support = self.bundle["support"]
        return {
            "categories": support["categories"],
            "make_models": support["make_models"],
            "numeric_ranges": support["numeric"],
            "field_rules": {
                "engine_size": {"min": 0.0, "max": 5.7, "step": 0.1, "nullable": True},
            },
            "units": {
                "mileage": "mile",
                "torque": "lb-ft",
                "fuel_efficiency": "MPG/MPGe",
                "currency": "USD",
            },
            "reference_year": self.bundle["reference_year"],
            "schema_version": self.bundle["schema_version"],
            "model_version": self.bundle["model_version"],
        }

    def _support(self, payload: VehicleInput) -> tuple[str, list[str]]:
        assert self.bundle
        support = self.bundle["support"]
        warnings: list[str] = []
        values = payload.model_dump()

        field_map = {
            "make": "Make",
            "model": "Model",
            "fuel_type": "Fuel_Type",
            "transmission": "Transmission",
            "service_history": "Service_History",
            "color": "Color",
            "body_type": "Body_Type",
            "drivetrain": "Drivetrain",
            "location": "Location",
        }
        for request_field, training_field in field_map.items():
            value = values[request_field]
            if value is not None and value not in support["categories"][training_field]:
                warnings.append(f"{training_field} was not observed in training data.")

        if (
            payload.make in support["make_models"]
            and payload.model not in support["make_models"][payload.make]
        ):
            warnings.append("This Make-Model pairing was not observed in training data.")

        numeric_map = {
            "year": "Year",
            "engine_size": "Engine_Size",
            "mileage": "Mileage",
            "horsepower": "Horsepower",
            "torque": "Torque",
            "owners": "Owners",
            "fuel_efficiency": "Fuel_Efficiency",
        }
        for request_field, training_field in numeric_map.items():
            value = values[request_field]
            if value is None:
                continue
            bounds = support["numeric"][training_field]
            if value < bounds["min"] or value > bounds["max"]:
                warnings.append(f"{training_field} is outside the observed training range.")

        if payload.engine_size == 0 and payload.fuel_type != "Electric":
            warnings.append(
                "Engine_Size 0.0 was treated as low-support for a non-electric vehicle."
            )
        return ("low_confidence" if warnings else "in_distribution", warnings)

    def predict(self, payload: VehicleInput) -> dict[str, Any]:
        if not self.bundle:
            raise RuntimeError(self.ready_error() or "Model unavailable")

        row = pd.DataFrame(
            [{
                "Make": payload.make,
                "Model": payload.model,
                "Year": payload.year,
                "Fuel_Type": payload.fuel_type,
                "Transmission": payload.transmission,
                "Engine_Size": payload.engine_size,
                "Mileage": payload.mileage,
                "Horsepower": payload.horsepower,
                "Torque": payload.torque,
                "Owners": payload.owners,
                "Accident_History": payload.accident_history,
                "Service_History": payload.service_history,
                "Color": payload.color,
                "Body_Type": payload.body_type,
                "Drivetrain": payload.drivetrain,
                "Fuel_Efficiency": payload.fuel_efficiency,
                "Location": payload.location,
            }]
        )
        x = build_features(row, self.bundle["reference_year"])
        raw = float(self.bundle["model"].predict(x)[0])
        estimate = float(np.expm1(raw) if self.bundle["objective"] == "log1p" else raw)
        if not np.isfinite(estimate):
            raise RuntimeError("Model produced a non-finite estimate.")
        estimate = max(0.0, estimate)

        interval = self.bundle["interval"]
        bucket = int(
            np.digitize([estimate], np.asarray(interval["edges"][1:-1], dtype=float))[0]
        )
        half_width = float(interval["quantiles"][bucket])
        if not np.isfinite(half_width) or half_width < 0:
            raise RuntimeError("Model interval calibration is invalid.")

        support, warnings = self._support(payload)
        if support == "low_confidence":
            half_width *= 1.35

        lower = max(0.0, estimate - half_width)
        upper = estimate + half_width

        top_factors: list[dict[str, str]] = []
        if self._explainer is not None:
            shap_values = self._explainer.shap_values(x)
            arr = np.asarray(shap_values)[0]
            order = np.argsort(np.abs(arr))[::-1][:3]
            for index in order:
                top_factors.append({
                    "feature": str(x.columns[index]),
                    "direction": "up" if float(arr[index]) >= 0 else "down",
                })

        return {
            "estimate": round(estimate, 2),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
            "support": support,
            "warnings": warnings,
            "top_factors": top_factors,
        }


runtime = Runtime()
