import numpy as np
import pandas as pd
import pytest

import ml.training.train as training
from ml.training.train import fit_mondrian_interval, interval_half_width, regression_metrics


def test_mondrian_handles_tied_predictions_without_empty_bucket_failure():
    predictions = np.ones(20) * 1000.0
    actuals = np.linspace(900.0, 1100.0, 20)
    interval = fit_mondrian_interval(predictions, actuals, bins=4, alpha=0.20)
    widths = interval_half_width(interval, np.array([1000.0, 1200.0]))
    assert len(interval["quantiles"]) == 4
    assert np.isfinite(widths).all()
    assert (widths >= 0).all()


def test_mondrian_rejects_empty_or_misaligned_inputs():
    with pytest.raises(ValueError):
        fit_mondrian_interval(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        fit_mondrian_interval(np.array([1.0, 2.0]), np.array([1.0]))


def test_regression_metrics_reject_non_finite_predictions():
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0]), np.array([np.inf]))



def test_cold_start_receives_only_primary_training_partition(monkeypatch):
    frame = pd.DataFrame({"row_id": np.arange(8)})
    train_idx = np.array([0, 1, 2, 3])
    calibration_idx = np.array([4, 5])
    test_idx = np.array([6, 7])
    observed: dict[str, object] = {}

    def fake_cold_start(pool: pd.DataFrame, objective: str):
        observed["rows"] = pool["row_id"].tolist()
        observed["objective"] = objective
        return training.Metrics(1.0, 1.0, 0.5, 0.1), ["A|B"], 1

    monkeypatch.setattr(training, "cold_start_stress", fake_cold_start)

    _, _, heldout_rows, pool_rows = training.cold_start_from_primary_training_split(
        frame,
        train_idx,
        calibration_idx,
        test_idx,
        "log1p",
    )

    assert observed["rows"] == [0, 1, 2, 3]
    assert observed["objective"] == "log1p"
    assert heldout_rows == 1
    assert pool_rows == 4


def test_cold_start_partition_helper_rejects_overlapping_indices():
    frame = pd.DataFrame({"row_id": np.arange(4)})

    with pytest.raises(RuntimeError, match="partitions overlap"):
        training.cold_start_from_primary_training_split(
            frame,
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([3]),
            "raw",
        )
