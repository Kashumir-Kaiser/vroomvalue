from pathlib import Path
import joblib
import pytest


@pytest.mark.model
def test_saved_model_passed_release_gates():
    path = Path("models/auto_price.joblib")
    if not path.exists():
        pytest.fail("Train the model first: python -m ml.training.train")
    metrics = joblib.load(path)["metrics"]
    assert metrics["baseline_improvement"] >= 0.10
    assert 0.77 <= metrics["coverage_80"] <= 0.83