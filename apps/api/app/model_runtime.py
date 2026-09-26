from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
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

logger = logging.getLogger("vroomvalue.runtime")

MODEL_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", "models/auto_price.joblib"))
DATA_PATH = Path(os.getenv("DATASET_PATH", str(DATASET_PATH)))
DEFAULT_REFRESH_TTL_SECONDS = 1.0


class _UnsetSignature:
    pass


_UNSET_SIGNATURE = _UnsetSignature()


def _read_refresh_ttl_seconds() -> float:
    raw = os.getenv("RUNTIME_REFRESH_TTL_SECONDS")
    if raw is None:
        return DEFAULT_REFRESH_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid RUNTIME_REFRESH_TTL_SECONDS=%r; using %.1f seconds.",
            raw,
            DEFAULT_REFRESH_TTL_SECONDS,
        )
        return DEFAULT_REFRESH_TTL_SECONDS
    if not math.isfinite(value):
        logger.warning(
            "Non-finite RUNTIME_REFRESH_TTL_SECONDS=%r; using %.1f seconds.",
            raw,
            DEFAULT_REFRESH_TTL_SECONDS,
        )
        return DEFAULT_REFRESH_TTL_SECONDS
    return max(0.0, value)


MODEL_LOAD_FAILURE_LOG_INTERVAL_SECONDS = 60.0

CATEGORICAL_SUPPORT_FIELDS = {
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
NUMERIC_SUPPORT_FIELDS = {
    "year": "Year",
    "engine_size": "Engine_Size",
    "mileage": "Mileage",
    "horsepower": "Horsepower",
    "torque": "Torque",
    "owners": "Owners",
    "fuel_efficiency": "Fuel_Efficiency",
}


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


class Runtime:
    def __init__(self, refresh_ttl_seconds: float | None = None) -> None:
        self.bundle: dict[str, Any] | None = None
        self.dataset_error: str | None = None
        self.model_error: str | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._dataset_signature: tuple[int, int, int] | None | _UnsetSignature = (
            _UNSET_SIGNATURE
        )
        self._model_signature: tuple[int, int, int] | None | _UnsetSignature = (
            _UNSET_SIGNATURE
        )
        self._refresh_ttl_seconds = (
            _read_refresh_ttl_seconds()
            if refresh_ttl_seconds is None
            else max(0.0, float(refresh_ttl_seconds))
        )
        self._last_refresh_check: float | None = None
        self._model_load_failure_count = 0
        self._last_model_failure_log_at: float | None = None
        self._lock = threading.RLock()

    def _load_dataset_gate(self, *, force: bool = False) -> None:
        signature = _file_signature(DATA_PATH)
        if (
            not force
            and self._dataset_signature is not _UNSET_SIGNATURE
            and signature == self._dataset_signature
        ):
            return
        try:
            validate_dataset_contract(DATA_PATH)
            self.dataset_error = None
        except DatasetContractError as exc:
            self.dataset_error = str(exc)
        self._dataset_signature = signature

    def _record_model_load_failure(
        self,
        exc: Exception,
        signature: tuple[int, int, int],
    ) -> None:
        self._model_load_failure_count += 1
        now = time.monotonic()
        if (
            self._last_model_failure_log_at is not None
            and now - self._last_model_failure_log_at
            < MODEL_LOAD_FAILURE_LOG_INTERVAL_SECONDS
        ):
            return

        self._last_model_failure_log_at = now
        logger.error(
            json.dumps(
                {
                    "event": "model.load_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "artifact_signature": signature,
                    "consecutive_failures": self._model_load_failure_count,
                    "retry_after_seconds": self._refresh_ttl_seconds,
                }
            )
        )

    def _record_model_load_recovery(self) -> None:
        if self._model_load_failure_count == 0:
            return
        logger.info(
            json.dumps(
                {
                    "event": "model.load_recovered",
                    "prior_consecutive_failures": self._model_load_failure_count,
                }
            )
        )
        self._model_load_failure_count = 0
        self._last_model_failure_log_at = None

    def _load_model(self, *, force: bool = False) -> None:
        signature = _file_signature(MODEL_PATH)
        if (
            not force
            and self._model_signature is not _UNSET_SIGNATURE
            and signature == self._model_signature
        ):
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
                "model",
                "objective",
                "reference_year",
                "interval",
                "support",
                "feature_columns",
                "model_name",
                "model_version",
                "schema_version",
                "as_of_date",
            }
            missing = sorted(required.difference(bundle))
            if missing:
                raise ValueError(f"artifact missing keys: {', '.join(missing)}")
            if not bundle["feature_columns"]:
                raise ValueError("artifact feature_columns is empty")

            self.bundle = bundle
            self.model_error = None
            self._record_model_load_recovery()
            try:
                self._explainer = shap.TreeExplainer(bundle["model"])
            except Exception:
                self._explainer = None
                logger.exception("SHAP explainer initialization failed; pricing remains available.")
        except Exception as exc:  # noqa: BLE001 - fail closed on artifact-load failures
            self.bundle = None
            self._explainer = None
            self.model_error = f"Model artifact could not be loaded: {exc}"
            self._record_model_load_failure(exc, signature)
            # Do not cache a failed non-missing artifact signature. A transient
            # partial write/read failure is retried after the refresh TTL even
            # when filesystem metadata happens to remain unchanged.
            self._model_signature = _UNSET_SIGNATURE
            return
        self._model_signature = signature

    def load(self, *, force: bool = False) -> None:
        with self._lock:
            self._load_dataset_gate(force=force)
            self._load_model(force=force)
            self._last_refresh_check = time.monotonic()

    def refresh_if_changed(self) -> None:
        now = time.monotonic()
        with self._lock:
            if (
                self._last_refresh_check is not None
                and now - self._last_refresh_check < self._refresh_ttl_seconds
            ):
                return
            self._last_refresh_check = now
            self._load_dataset_gate(force=False)
            self._load_model(force=False)

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

        for request_field, training_field in CATEGORICAL_SUPPORT_FIELDS.items():
            value = getattr(payload, request_field)
            if value is not None and value not in support["categories"][training_field]:
                warnings.append(f"{training_field} was not observed in training data.")

        if (
            payload.make in support["make_models"]
            and payload.model not in support["make_models"][payload.make]
        ):
            warnings.append("This Make-Model pairing was not observed in training data.")

        for request_field, training_field in NUMERIC_SUPPORT_FIELDS.items():
            value = getattr(payload, request_field)
            if value is None:
                continue
            bounds = support["numeric"][training_field]
            if value < bounds["min"] or value > bounds["max"]:
                warnings.append(f"{training_field} is outside the observed training range.")

        if payload.engine_size == 0 and payload.fuel_type.casefold() != "electric":
            warnings.append(
                "Engine_Size 0.0 was treated as low-support for a non-electric vehicle."
            )
        return ("low_confidence" if warnings else "in_distribution", warnings)

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.bundle:
            raise RuntimeError(self.ready_error() or "Model unavailable")

        expected = list(self.bundle["feature_columns"])
        actual = list(features.columns)
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise RuntimeError(
                "Inference feature schema mismatch: " + "; ".join(details)
            )
        return features.reindex(columns=expected)

    def _explain(
        self,
        features: pd.DataFrame,
        warnings: list[str],
    ) -> list[dict[str, str]]:
        if self._explainer is None:
            warnings.append("Explanation unavailable for this prediction.")
            return []

        try:
            shap_values = self._explainer.shap_values(features)
            values = np.asarray(shap_values)
            if values.ndim == 1:
                row_values = values
            elif values.ndim >= 2:
                row_values = values[0]
            else:
                raise ValueError("SHAP returned an unexpected value shape")

            if len(row_values) != len(features.columns):
                raise ValueError("SHAP feature count does not match inference features")

            order = np.argsort(np.abs(row_values))[::-1][:3]
            return [
                {
                    "feature": str(features.columns[index]),
                    "direction": "up" if float(row_values[index]) >= 0 else "down",
                }
                for index in order
            ]
        except Exception:
            logger.exception("SHAP explanation failed for a prediction.")
            warnings.append("Explanation unavailable for this prediction.")
            return []

    def predict(self, payload: VehicleInput) -> dict[str, Any]:
        if not self.bundle:
            raise RuntimeError(self.ready_error() or "Model unavailable")

        row = pd.DataFrame(
            [
                {
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
                }
            ]
        )
        features = self._align_features(
            build_features(row, self.bundle["reference_year"])
        )
        raw = float(self.bundle["model"].predict(features)[0])
        estimate = float(
            np.expm1(raw) if self.bundle["objective"] == "log1p" else raw
        )
        if not np.isfinite(estimate):
            raise RuntimeError("Model produced a non-finite estimate.")
        estimate = max(0.0, estimate)

        interval = self.bundle["interval"]
        bucket = int(
            np.digitize(
                [estimate],
                np.asarray(interval["edges"][1:-1], dtype=float),
            )[0]
        )
        half_width = float(interval["quantiles"][bucket])
        if not np.isfinite(half_width) or half_width < 0:
            raise RuntimeError("Model interval calibration is invalid.")

        support, warnings = self._support(payload)
        if support == "low_confidence":
            half_width *= 1.35

        lower = max(0.0, estimate - half_width)
        upper = estimate + half_width
        top_factors = self._explain(features, warnings)

        return {
            "estimate": round(estimate, 2),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
            "support": support,
            "warnings": warnings,
            "top_factors": top_factors,
        }


runtime = Runtime()
