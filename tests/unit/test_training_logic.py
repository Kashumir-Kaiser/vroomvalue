import numpy as np
import pytest

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
