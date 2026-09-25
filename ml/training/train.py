from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ml.contracts.schema import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    PARTIALLY_MISSING_COLUMNS,
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
    denom = float(np.abs(y_true).sum())
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
        wape=float(np.abs(y_true - y_pred).sum() / denom) if denom else math.nan,
    )


def _four_way_group_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return train, calibration, test


def _predict(model: CatBoostRegressor, x: pd.DataFrame, objective: str) -> np.ndarray:
    pred = model.predict(x)
    return np.expm1(pred) if objective == "log1p" else pred


def _fit_model(frame: pd.DataFrame, objective: str) -> CatBoostRegressor:
    x = build_features(frame, REFERENCE_YEAR)
    y = frame["Selling_Price"].to_numpy(dtype=float)
    if objective == "log1p":
        y = np.log1p(y)
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(x, y, cat_features=CATEGORICAL_COLUMNS)
    return model


def choose_objective_by_group_cv(frame: pd.DataFrame) -> tuple[str, dict[str, float]]:
    groups = duplicate_fingerprint(frame)
    folds = GroupKFold(n_splits=5)
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


def fit_mondrian_interval(
    calibration_pred: np.ndarray,
    calibration_y: np.ndarray,
    bins: int = 4,
    alpha: float = 0.20,
) -> dict[str, Any]:
    edges = np.quantile(calibration_pred, np.linspace(0, 1, bins + 1)).astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    residuals = np.abs(calibration_y - calibration_pred)
    quantiles: list[float] = []
    for i in range(bins):
        mask = (calibration_pred >= edges[i]) & (calibration_pred < edges[i + 1])
        bucket = residuals[mask]
        n = len(bucket)
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        quantiles.append(float(np.quantile(bucket, level, method="higher")))
    return {"edges": edges.tolist(), "quantiles": quantiles, "alpha": alpha}


def interval_half_width(interval: dict[str, Any], predictions: np.ndarray) -> np.ndarray:
    inner_edges = np.asarray(interval["edges"][1:-1], dtype=float)
    buckets = np.digitize(predictions, inner_edges)
    q = np.asarray(interval["quantiles"], dtype=float)
    return q[buckets]


def model_support(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = {}
    for col in ["Year", "Engine_Size", "Mileage", "Horsepower", "Torque", "Owners", "Fuel_Efficiency"]:
        series = frame[col].dropna().astype(float)
        numeric[col] = {"min": float(series.min()), "max": float(series.max())}
    return {
        "categories": category_metadata(frame),
        "make_models": make_model_map(frame),
        "numeric": numeric,
    }


def write_model_card(
    objective: str,
    cv_mae: dict[str, float],
    baseline_metrics: Metrics,
    model_metrics: Metrics,
    coverage: float,
    interval: dict[str, Any],
    improvement: float,
) -> None:
    content = f"""# VroomValue model card

## Purpose

VroomValue estimates a used vehicle's **Selling_Price** from the supplied 17 predictors. It is decision assistance, not an appraisal guarantee. Displayed amounts use **USD**, mileage uses **miles**, torque uses **lb-ft**, and fuel efficiency uses **MPG / MPGe for electric vehicles** because the source data does not provide explicit unit columns.

## Data

- Source: `data/automobile_dataset.csv` supplied with this project.
- Rows: 5,500.
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

Intervals use split-conformal absolute residuals with four prediction-price buckets (Mondrian calibration). This keeps the uncertainty width more appropriate across low- and high-price vehicles. Bucket half-widths are: {', '.join(f'${q:,.0f}' for q in interval['quantiles'])}.

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
    train_idx, calibration_idx, test_idx = _four_way_group_split(frame)
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
        "as_of_date": str(date.today()),
        "metrics": {
            "baseline": asdict(baseline_metrics),
            "holdout": asdict(model_metrics),
            "baseline_improvement": improvement,
            "coverage_80": coverage,
            "cv_mae": cv_mae,
        },
    }
    joblib.dump(bundle, artifact_path)

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
        objective, cv_mae, baseline_metrics, model_metrics, coverage, interval, improvement
    )

    if not args.skip_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise SystemExit("MLflow is required for release training. Install project dependencies and retry.") from exc
        mlflow.set_tracking_uri("file:./mlruns")
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
                }
            )
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            mlflow.log_artifact(str(MODEL_CARD_PATH))
            mlflow.log_artifact(str(SPLIT_MANIFEST_PATH))

    print(json.dumps(bundle["metrics"], indent=2))
    print(f"Saved champion artifact to {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())