from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATASET_PATH = Path("data/automobile_dataset.csv")
EXPECTED_COLUMNS = [
    "Make", "Model", "Year", "Fuel_Type", "Transmission", "Engine_Size", "Mileage",
    "Horsepower", "Torque", "Owners", "Accident_History", "Service_History", "Color",
    "Body_Type", "Drivetrain", "Fuel_Efficiency", "Location", "Selling_Price",
]
PARTIALLY_MISSING_COLUMNS = [
    "Transmission", "Engine_Size", "Horsepower", "Torque", "Accident_History",
    "Service_History", "Color", "Fuel_Efficiency", "Location",
]
CATEGORICAL_COLUMNS = [
    "Make", "Model", "Fuel_Type", "Transmission", "Service_History", "Color",
    "Body_Type", "Drivetrain", "Location",
]
NUMERIC_COLUMNS = [
    "Year", "Engine_Size", "Mileage", "Horsepower", "Torque", "Owners",
    "Accident_History", "Fuel_Efficiency",
]
TARGET_COLUMN = "Selling_Price"
REFERENCE_YEAR = 2026
ENGINE_SIZE_MIN = 0.0
ENGINE_SIZE_MAX = 5.7
ENGINE_SIZE_STEP = 0.1


class DatasetContractError(RuntimeError):
    pass


def _one_decimal_or_less(values: pd.Series) -> pd.Series:
    non_null = values.dropna().astype(float)
    scaled = non_null * 10
    return np.isclose(scaled, np.round(scaled), atol=1e-9)


def validate_dataset_contract(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetContractError(
            f"Dataset not found at {path.as_posix()}. Add the file and retry."
        )
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetContractError(
            f"Dataset at {path.as_posix()} contains no rows. Add data and retry."
        ) from exc

    if frame.empty:
        raise DatasetContractError(
            f"Dataset at {path.as_posix()} contains no rows. Add data and retry."
        )

    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetContractError(
            "Dataset at "
            f"{path.as_posix()} is missing expected columns: {', '.join(missing)}."
        )

    frame = frame[EXPECTED_COLUMNS].copy()
    problems: list[str] = []
    if not frame["Year"].between(2005, 2024).all():
        problems.append("Year must be between 2005 and 2024")
    if (frame["Mileage"] < 0).any():
        problems.append("Mileage must be non-negative")
    if not frame["Owners"].between(1, 5).all():
        problems.append("Owners must be between 1 and 5")
    accident = frame["Accident_History"].dropna()
    if not accident.isin([0, 1]).all():
        problems.append("Accident_History must be 0, 1, or missing")

    engine = frame["Engine_Size"].dropna()
    if not engine.between(ENGINE_SIZE_MIN, ENGINE_SIZE_MAX).all():
        problems.append("Engine_Size must be between 0.0 and 5.7")
    if not _one_decimal_or_less(frame["Engine_Size"]).all():
        problems.append("Engine_Size must use at most one decimal place (0.1 increments)")

    if problems:
        raise DatasetContractError("; ".join(problems))
    return frame


def category_metadata(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        col: sorted(frame[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_COLUMNS
    }


def make_model_map(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        make: sorted(group["Model"].astype(str).unique().tolist())
        for make, group in frame.groupby("Make", sort=True)
    }


def duplicate_fingerprint(frame: pd.DataFrame) -> pd.Series:
    columns: Iterable[str] = [
        "Make", "Model", "Year", "Mileage", "Engine_Size", "Horsepower", "Torque",
        "Location", "Selling_Price",
    ]
    normalized = frame[list(columns)].copy()
    for col in normalized.select_dtypes(include="object"):
        normalized[col] = normalized[col].fillna("MISSING").str.strip().str.upper()
    return normalized.fillna("<NA>").astype(str).agg("|".join, axis=1)