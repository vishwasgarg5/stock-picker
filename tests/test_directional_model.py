import numpy as np
import pandas as pd

from src.directional_model import _recent


def test_recent_requires_full_window():
    x = pd.DataFrame({
        "model_accuracy_pct": [55.0, 56.0, 54.0],
        "accuracy_lift_pct": [2.0, 1.0, 3.0],
    })
    out = _recent(x)
    assert out["recent_5_sessions"] == 3
    assert np.isnan(out["recent_5_lift_pct"])


def test_recent_full_window_is_available():
    x = pd.DataFrame({
        "model_accuracy_pct": [55.0] * 20,
        "accuracy_lift_pct": [1.0] * 20,
    })
    out = _recent(x)
    assert out["recent_20_sessions"] == 20
    assert out["recent_20_lift_pct"] == 1.0
