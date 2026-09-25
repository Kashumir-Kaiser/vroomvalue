from pathlib import Path
import pandas as pd
import pytest
from ml.contracts.schema import DatasetContractError, EXPECTED_COLUMNS, validate_dataset_contract


def test_real_dataset_contract():
    df = validate_dataset_contract("data/automobile_dataset.csv")
    assert len(df) == 5500
    assert list(df.columns) == EXPECTED_COLUMNS
    assert ((df["Engine_Size"].dropna() * 10).round() == df["Engine_Size"].dropna() * 10).all()


def test_missing_file_exact_message(tmp_path: Path):
    p = tmp_path / "automobile_dataset.csv"
    with pytest.raises(DatasetContractError, match="Dataset not found") as exc:
        validate_dataset_contract(p)
    assert str(exc.value) == f"Dataset not found at {p.as_posix()}. Add the file and retry."


def test_empty_file_exact_message(tmp_path: Path):
    p = tmp_path / "automobile_dataset.csv"
    p.write_text(",".join(EXPECTED_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(DatasetContractError) as exc:
        validate_dataset_contract(p)
    assert str(exc.value) == f"Dataset at {p.as_posix()} contains no rows. Add data and retry."


def test_missing_column_names_it(tmp_path: Path):
    p = tmp_path / "automobile_dataset.csv"
    pd.DataFrame([{c: 1 for c in EXPECTED_COLUMNS if c != "Torque"}]).to_csv(p, index=False)
    with pytest.raises(DatasetContractError) as exc:
        validate_dataset_contract(p)
    assert "Torque" in str(exc.value)


def test_engine_size_rejects_more_than_one_decimal(tmp_path: Path):
    base = validate_dataset_contract("data/automobile_dataset.csv").head(1)
    base.loc[base.index[0], "Engine_Size"] = 2.55
    p = tmp_path / "automobile_dataset.csv"
    base.to_csv(p, index=False)
    with pytest.raises(DatasetContractError, match="at most one decimal place"):
        validate_dataset_contract(p)

def test_required_missing_value_is_rejected(tmp_path: Path):
    base = validate_dataset_contract("data/automobile_dataset.csv").head(1)
    base.loc[base.index[0], "Mileage"] = float("nan")
    p = tmp_path / "automobile_dataset.csv"
    base.to_csv(p, index=False)
    with pytest.raises(DatasetContractError, match="Required columns contain missing values"):
        validate_dataset_contract(p)


def test_non_finite_numeric_is_rejected(tmp_path: Path):
    base = validate_dataset_contract("data/automobile_dataset.csv").head(1)
    base.loc[base.index[0], "Horsepower"] = float("inf")
    p = tmp_path / "automobile_dataset.csv"
    base.to_csv(p, index=False)
    with pytest.raises(DatasetContractError, match="finite numeric values"):
        validate_dataset_contract(p)


def test_numeric_parse_failure_is_explicit(tmp_path: Path):
    base = validate_dataset_contract("data/automobile_dataset.csv").head(1)
    base.loc[base.index[0], "Torque"] = "not-a-number"
    p = tmp_path / "automobile_dataset.csv"
    base.to_csv(p, index=False)
    with pytest.raises(DatasetContractError, match="Numeric parse failures.*Torque"):
        validate_dataset_contract(p)
