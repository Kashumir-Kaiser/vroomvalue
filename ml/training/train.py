from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ml.artifacts import write_checksum
from ml.contracts.schema import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    REFERENCE_YEAR,
    category_metadata,
    duplicate_fingerprint,
    make_model_map,
    validate_dataset_contract,
)
from ml.features.build import build_features

RANDOM_SEED = 42
ARTIFACT_PATH = Path("models/auto_price.joblib")
SPLIT_MANIFEST_PATH = Path("models/split_manifest.json")
MODEL_CARD_PATH = Path("model_card.md")
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

def _mlflow_tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    return configured or DEFAULT_MLFLOW_TRACKING_URI


CATBOOST_PARAMS = {
    "iterations": 350,
    "depth": 7,
    "learning_rate": 0.06,
    "loss_function": "MAE",
    "verbose": False,
    "random_seed": RANDOM_SEED,
    "l2_leaf_reg": 5,
    "random_strength": 0.3,
    "allow_writing_files": False,
    "thread_count": 4,
}


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    wape: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Metric inputs must be non-empty and have equal length.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Metric inputs must contain only finite values.")
    denom = float(np.abs(y_true).sum())
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
        wape=float(np.abs(y_true - y_pred).sum() / denom) if denom else math.nan,
    )


def _three_way_group_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    groups = duplicate_fingerprint(frame)
    first = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_SEED)
    train_cal_pos, test_pos = next(first.split(indices, groups=groups))
    train_cal = indices[train_cal_pos]
    test = indices[test_pos]

    second = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=RANDOM_SEED + 1)
    train_pos, calibration_pos = next(
        second.split(train_cal, groups=groups.iloc[train_cal])
    )
    train = train_cal[train_pos]
    calibration = train_cal[calibration_pos]

    train_groups = set(groups.iloc[train])
    calibration_groups = set(groups.iloc[calibration])
    test_groups = set(groups.iloc[test])
    if train_groups & calibration_groups or train_groups & test_groups or calibration_groups & test_groups:
        raise RuntimeError("Duplicate-group leakage detected across evaluation splits.")
    return train, calibration, test


def _predict(model: CatBoostRegressor, x: pd.DataFrame, objective: str) -> np.ndarray:
    pred = np.asarray(model.predict(x), dtype=float)
    out = np.expm1(pred) if objective == "log1p" else pred
    if not np.isfinite(out).all():
        raise RuntimeError("Model produced non-finite predictions.")
    return out


def _fit_model(frame: pd.DataFrame, objective: str) -> CatBoostRegressor:
    if frame.empty:
        raise ValueError("Cannot train on an empty frame.")
    x = build_features(frame, REFERENCE_YEAR)
    y = frame["Selling_Price"].to_numpy(dtype=float)
    if objective == "log1p":
        y = np.log1p(y)
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(x, y, cat_features=CATEGORICAL_COLUMNS)
    return model


def choose_objective_by_group_cv(frame: pd.DataFrame) -> tuple[str, dict[str, float]]:
    groups = duplicate_fingerprint(frame)
    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Too few duplicate groups for grouped cross-validation.")
    folds = GroupKFold(n_splits=n_splits)
    scores: dict[str, list[float]] = {"raw": [], "log1p": []}
    for train_pos, valid_pos in folds.split(frame, groups=groups):
        train = frame.iloc[train_pos]
        valid = frame.iloc[valid_pos]
        x_valid = build_features(valid, REFERENCE_YEAR)
        y_valid = valid["Selling_Price"].to_numpy(dtype=float)
        for objective in ("raw", "log1p"):
            model = _fit_model(train, objective)
            pred = _predict(model, x_valid, objective)
            scores[objective].append(float(mean_absolute_error(y_valid, pred)))
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    return min(means, key=means.get), means


def fit_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "make_model_year": frame.groupby(["Make", "Model", "Year"])["Selling_Price"].median().to_dict(),
        "make_model": frame.groupby(["Make", "Model"])["Selling_Price"].median().to_dict(),
        "make": frame.groupby("Make")["Selling_Price"].median().to_dict(),
        "global": float(frame["Selling_Price"].median()),
    }


def predict_baseline(baseline: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    out: list[float] = []
    for row in frame.itertuples(index=False):
        value = baseline["make_model_year"].get((row.Make, row.Model, row.Year))
        if value is None:
            value = baseline["make_model"].get((row.Make, row.Model))
        if value is None:
            value = baseline["make"].get(row.Make, baseline["global"])
        out.append(float(value))
    return np.asarray(out)


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("Cannot calibrate an interval from zero residuals.")
    n = residuals.size
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def fit_mondrian_interval(
    calibration_pred: np.ndarray,
    calibration_y: np.ndarray,
    bins: int = 4,
    alpha: float = 0.20,
) -> dict[str, Any]:
    calibration_pred = np.asarray(calibration_pred, dtype=float)
    calibration_y = np.asarray(calibration_y, dtype=float)
    if calibration_pred.shape != calibration_y.shape or calibration_pred.size == 0:
        raise ValueError("Calibration predictions and labels must be non-empty and aligned.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if bins < 1:
        raise ValueError("bins must be at least 1.")

    residuals = np.abs(calibration_y - calibration_pred)
    global_q = _conformal_quantile(residuals, alpha)
    edges = np.quantile(calibration_pred, np.linspace(0, 1, bins + 1)).astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    quantiles: list[float] = []
    bucket_sizes: list[int] = []
    for i in range(bins):
        is_last = i == bins - 1
        right = calibration_pred <= edges[i + 1] if is_last else calibration_pred < edges[i + 1]
        mask = (calibration_pred >= edges[i]) & right
        bucket = residuals[mask]
        bucket_sizes.append(int(bucket.size))
        quantiles.append(_conformal_quantile(bucket, alpha) if bucket.size else global_q)
    return {
        "edges": edges.tolist(),
        "quantiles": quantiles,
        "bucket_sizes": bucket_sizes,
        "alpha": alpha,
        "fallback_global_quantile": global_q,
    }


def interval_half_width(interval: dict[str, Any], predictions: np.ndarray) -> np.ndarray:
    inner_edges = np.asarray(interval["edges"][1:-1], dtype=float)
    buckets = np.digitize(predictions, inner_edges)
    q = np.asarray(interval["quantiles"], dtype=float)
    if q.size == 0 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Interval quantiles are invalid.")
    return q[buckets]


def model_support(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = {}
    for col in ["Year", "Engine_Size", "Mileage", "Horsepower", "Torque", "Owners", "Fuel_Efficiency"]:
        series = frame[col].dropna().astype(float)
        if series.empty:
            continue
        numeric[col] = {"min": float(series.min()), "max": float(series.max())}
    return {
        "categories": category_metadata(frame),
        "make_models": make_model_map(frame),
        "numeric": numeric,
    }


def cold_start_stress(
    frame: pd.DataFrame, objective: str
) -> tuple[Metrics, list[str], int]:
    groups = frame["Make"].astype(str) + "|" + frame["Model"].astype(str)
    if groups.nunique() < 2:
        raise ValueError("Too few Make-Model groups for cold-start evaluation.")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_SEED + 2
    )
    train_pos, test_pos = next(splitter.split(frame, groups=groups))
    train = frame.iloc[train_pos]
    test = frame.iloc[test_pos]
    heldout_groups = sorted(groups.iloc[test_pos].unique().tolist())
    if set(groups.iloc[train_pos]) & set(heldout_groups):
        raise RuntimeError("Make-Model leakage detected in cold-start stress split.")
    model = _fit_model(train, objective)
    pred = _predict(model, build_features(test, REFERENCE_YEAR), objective)
    metrics = regression_metrics(test["Selling_Price"].to_numpy(dtype=float), pred)
    return metrics, heldout_groups, len(test)


def cold_start_from_primary_training_split(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
    objective: str,
) -> tuple[Metrics, list[str], int, int]:
    """Run cold-start stress only on the primary training partition."""
    train_ids = set(np.asarray(train_idx, dtype=int).tolist())
    calibration_ids = set(np.asarray(calibration_idx, dtype=int).tolist())
    test_ids = set(np.asarray(test_idx, dtype=int).tolist())
    if (
        train_ids & calibration_ids
        or train_ids & test_ids
        or calibration_ids & test_ids
    ):
        raise RuntimeError("Primary evaluation partitions overlap.")

    pool = frame.iloc[np.asarray(train_idx, dtype=int)].copy()
    metrics, groups, heldout_rows = cold_start_stress(pool, objective)
    return metrics, groups, heldout_rows, len(pool)


def write_model_card(
    dataset_rows: int,
    cold_pool_rows: int,
    objective: str,
    cv_mae: dict[str, float],
    baseline_metrics: Metrics,
    model_metrics: Metrics,
    coverage: float,
    interval: dict[str, Any],
    improvement: float,
    cold_metrics: Metrics,
    cold_groups: list[str],
    cold_rows: int,
) -> None:
    content = f"""# VroomValue model card

## Purpose

VroomValue estimates a used vehicle's **Selling_Price** from the supplied 17 predictors. It is decision assistance, not an appraisal guarantee. Displayed amounts use **USD**, mileage uses **miles**, torque uses **lb-ft**, and fuel efficiency uses **MPG / MPGe for electric vehicles** because the source data does not provide explicit unit columns.

## Data

- Source: `data/automobile_dataset.csv` supplied with this project.
- Rows: {dataset_rows:,}.
- Target: `Selling_Price`.
- Reference year frozen into the artifact: {REFERENCE_YEAR}.
- `Engine_Size` contract: 0.0–5.7 and at most one decimal place (0.1 increments); 0.0 is valid for electric vehicles.
- Missing values are preserved for numeric CatBoost inputs and represented as explicit `MISSING` categories for categorical fields, with missing indicators for each partially missing field.

## Evaluation design

Duplicate fingerprints are grouped before splitting. Ten percent is a locked interpolation holdout and ten percent is a dedicated calibration set. Raw-price and `log1p(price)` objectives are compared using 5-fold grouped CV on the development rows. The calibration set is not used to choose the point model.

## Current local training result

- Selected objective: **{objective}**
- Grouped CV MAE — raw: **${cv_mae['raw']:,.2f}**; log1p: **${cv_mae['log1p']:,.2f}**
- Naive Make-Model-Year baseline MAE: **${baseline_metrics.mae:,.2f}**
- CatBoost holdout MAE: **${model_metrics.mae:,.2f}**
- MAE improvement over baseline: **{improvement * 100:.1f}%**
- Holdout RMSE: **${model_metrics.rmse:,.2f}**
- Holdout R²: **{model_metrics.r2:.4f}**
- Holdout WAPE: **{model_metrics.wape * 100:.2f}%**
- Nominal 80% interval holdout coverage: **{coverage * 100:.2f}%**

### Cold-start Make-Model stress view

This stress test is derived only from the main training partition; calibration and locked holdout rows are excluded.

- Source training-pool rows: **{cold_pool_rows:,}**
- Held-out rows: **{cold_rows:,}**
- Held-out Make-Model groups: **{', '.join(cold_groups)}**
- Cold-start MAE: **${cold_metrics.mae:,.2f}**
- Cold-start WAPE: **{cold_metrics.wape * 100:.2f}%**

Intervals use split-conformal absolute residuals with four prediction-price buckets (Mondrian calibration). Empty buckets caused by tied predictions fall back to the global finite-sample conformal quantile. Bucket half-widths are: {', '.join(f'${q:,.0f}' for q in interval['quantiles'])}.

## Limitations

The dataset has no listing/sale date or source identifier, so this model cannot measure market-time drift or source bias. `Selling_Price` is treated as the target exactly as named in the supplied data; the source does not establish whether it is a listing price or a verified transaction price. Units/currency are assumptions and are stated in the UI. SHAP explanations describe model behavior, not causal price effects.
"""
    MODEL_CARD_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATASET_PATH))
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH))
    parser.add_argument("--skip-mlflow", action="store_true", help="Local harness only; release training must log to MLflow")
    args = parser.parse_args()

    frame = validate_dataset_contract(args.data)
    train_idx, calibration_idx, test_idx = _three_way_group_split(frame)
    train = frame.iloc[train_idx].copy()
    calibration = frame.iloc[calibration_idx].copy()
    test = frame.iloc[test_idx].copy()

    objective, cv_mae = choose_objective_by_group_cv(train)
    model = _fit_model(train, objective)

    calibration_pred = _predict(model, build_features(calibration, REFERENCE_YEAR), objective)
    interval = fit_mondrian_interval(
        calibration_pred,
        calibration["Selling_Price"].to_numpy(dtype=float),
    )

    test_x = build_features(test, REFERENCE_YEAR)
    test_y = test["Selling_Price"].to_numpy(dtype=float)
    test_pred = _predict(model, test_x, objective)
    model_metrics = regression_metrics(test_y, test_pred)

    baseline = fit_baseline(train)
    baseline_pred = predict_baseline(baseline, test)
    baseline_metrics = regression_metrics(test_y, baseline_pred)
    improvement = 1 - (model_metrics.mae / baseline_metrics.mae)

    half_width = interval_half_width(interval, test_pred)
    coverage = float(np.mean((test_y >= test_pred - half_width) & (test_y <= test_pred + half_width)))

    if improvement < 0.10:
        raise SystemExit(
            f"MODEL GATE FAILED: CatBoost MAE improvement is {improvement:.3%}; required >= 10%."
        )
    if not 0.77 <= coverage <= 0.83:
        raise SystemExit(
            f"MODEL GATE FAILED: 80% interval coverage is {coverage:.3%}; required 77%-83%."
        )

    cold_metrics, cold_groups, cold_rows, cold_pool_rows = (
        cold_start_from_primary_training_split(
            frame,
            train_idx,
            calibration_idx,
            test_idx,
            objective,
        )
    )

    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "objective": objective,
        "reference_year": REFERENCE_YEAR,
        "categorical_features": CATEGORICAL_COLUMNS,
        "feature_columns": build_features(train, REFERENCE_YEAR).columns.tolist(),
        "interval": interval,
        "support": model_support(train),
        "model_name": "auto_price",
        "model_version": "1",
        "schema_version": "1.0",
        "as_of_date": str(datetime.now(UTC).date()),
        "metrics": {
            "baseline": asdict(baseline_metrics),
            "holdout": asdict(model_metrics),
            "baseline_improvement": improvement,
            "coverage_80": coverage,
            "cv_mae": cv_mae,
            "cold_start": asdict(cold_metrics),
            "cold_start_pool_rows": cold_pool_rows,
            "cold_start_rows": cold_rows,
            "cold_start_groups": cold_groups,
        },
    }
    joblib.dump(bundle, artifact_path)
    checksum_path = write_checksum(artifact_path)

    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "seed": RANDOM_SEED,
                "train": train_idx.tolist(),
                "calibration": calibration_idx.tolist(),
                "test": test_idx.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_model_card(
        len(frame),
        cold_pool_rows,
        objective,
        cv_mae,
        baseline_metrics,
        model_metrics,
        coverage,
        interval,
        improvement,
        cold_metrics,
        cold_groups,
        cold_rows,
    )

    if not args.skip_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise SystemExit("MLflow is required for release training. Install project dependencies and retry.") from exc
        mlflow.set_tracking_uri(_mlflow_tracking_uri())
        mlflow.set_experiment("vroomvalue")
        with mlflow.start_run(run_name="catboost-champion"):
            mlflow.log_params({**CATBOOST_PARAMS, "objective_variant": objective, "reference_year": REFERENCE_YEAR})
            mlflow.log_metrics(
                {
                    "holdout_mae": model_metrics.mae,
                    "holdout_rmse": model_metrics.rmse,
                    "holdout_r2": model_metrics.r2,
                    "holdout_wape": model_metrics.wape,
                    "baseline_mae": baseline_metrics.mae,
                    "baseline_improvement": improvement,
                    "coverage_80": coverage,
                    "cold_start_mae": cold_metrics.mae,
                    "cold_start_wape": cold_metrics.wape,
                }
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            mlflow.log_artifact(str(checksum_path), artifact_path="model")
            mlflow.log_artifact(str(MODEL_CARD_PATH))
            mlflow.log_artifact(str(SPLIT_MANIFEST_PATH))

    print(json.dumps(bundle["metrics"], indent=2))
    print(f"Saved champion artifact to {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())