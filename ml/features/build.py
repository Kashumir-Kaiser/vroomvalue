from __future__ import annotations

import numpy as np
import pandas as pd

from ml.contracts.schema import CATEGORICAL_COLUMNS, PARTIALLY_MISSING_COLUMNS, REFERENCE_YEAR


def build_features(frame: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    x = frame.copy()
    if "Selling_Price" in x.columns:
        x = x.drop(columns=["Selling_Price"])

    x["vehicle_age"] = reference_year - x["Year"]
    x["mileage_per_year"] = x["Mileage"] / x["vehicle_age"].clip(lower=1)
    x["log_mileage"] = np.log1p(x["Mileage"])

    for column in PARTIALLY_MISSING_COLUMNS:
        x[f"{column}_missing"] = x[column].isna().astype("int8")

    for column in CATEGORICAL_COLUMNS:
        x[column] = x[column].fillna("MISSING").astype(str)

    return x