from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from apps.api.app.schemas import VehicleInput
from ml.contracts.schema import DATASET_PATH, DatasetContractError, validate_dataset_contract
from ml.features.build import build_features

MODEL_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", "models/auto_price.joblib"))
DATA_PATH = Path(os.getenv("DATASET_PATH", str(DATASET_PATH)))


class Runtime:
    def __init__(self) -> None:
        self.bundle: dict[str, Any] | None = None
        self.dataset_error: str | None = None
        self.model_error: str | None = None
        self._explainer: shap.TreeExplainer | None = None

    def load(self) -> None:
        try:
            validate_dataset_contract(DATA_PATH)
            self.dataset_error = None
        except DatasetContractError as exc:
            self.dataset_error = str(exc)

        if not MODEL_PATH.exists():
            self.model_error = f"Model artifact not found at {MODEL_PATH.as_posix()}. Train the model and retry."
            self.bundle = None
            return
        try:
            self.bundle = joblib.load(MODEL_PATH)
            self._explainer = shap.TreeExplainer(self.bundle["model"])
            self.model_error = None
        except Exception as exc:  # fail closed: readiness reports the exact load problem
            self.bundle = None
            self.model_error = f"Model artifact could not be loaded: {exc}"

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
            "units": {"mileage": "mile", "torque": "lb-ft", "fuel_efficiency": "MPG/MPGe", "currency": "USD"},
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
            "make": "Make", "model": "Model", "fuel_type": "Fuel_Type",
            "transmission": "Transmission", "service_history": "Service_History",
            "color": "Color", "body_type": "Body_Type", "drivetrain": "Drivetrain",
            "location": "Location",
        }
        for request_field, training_field in field_map.items():
            value = values[request_field]
            if value is not None and value not in support["categories"][training_field]:
                warnings.append(f"{training_field} was not observed in training data.")

        if payload.make in support["make_models"] and payload.model not in support["make_models"][payload.make]:
            warnings.append("This Make-Model pairing was not observed in training data.")

        numeric_map = {
            "year": "Year", "engine_size": "Engine_Size", "mileage": "Mileage",
            "horsepower": "Horsepower", "torque": "Torque", "owners": "Owners",
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
            warnings.append("Engine_Size 0.0 was treated as low-support for a non-electric vehicle.")
        return ("low_confidence" if warnings else "in_distribution", warnings)

    def predict(self, payload: VehicleInput) -> dict[str, Any]:
        if not self.bundle:
            raise RuntimeError(self.ready_error() or "Model unavailable")
        row = pd.DataFrame(
            [{
                "Make": payload.make, "Model": payload.model, "Year": payload.year,
                "Fuel_Type": payload.fuel_type, "Transmission": payload.transmission,
                "Engine_Size": payload.engine_size, "Mileage": payload.mileage,
                "Horsepower": payload.horsepower, "Torque": payload.torque,
                "Owners": payload.owners, "Accident_History": payload.accident_history,
                "Service_History": payload.service_history, "Color": payload.color,
                "Body_Type": payload.body_type, "Drivetrain": payload.drivetrain,
                "Fuel_Efficiency": payload.fuel_efficiency, "Location": payload.location,
            }]
        )
        x = build_features(row, self.bundle["reference_year"])
        raw = float(self.bundle["model"].predict(x)[0])
        estimate = float(np.expm1(raw) if self.bundle["objective"] == "log1p" else raw)
        estimate = max(0.0, estimate)

        interval = self.bundle["interval"]
        bucket = int(np.digitize([estimate], np.asarray(interval["edges"][1:-1], dtype=float))[0])
        half_width = float(interval["quantiles"][bucket])
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