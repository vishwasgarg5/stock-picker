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


def test_walk_forward_split_is_chronological():
    from src.directional_model import _walk_forward_splits
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    splits = list(_walk_forward_splits(list(dates)))
    assert splits
    train, test = splits[0]
    assert len(train) == 60
    assert len(test) == 20
    assert max(train) < min(test)


def test_meaningful_move_target_excludes_tiny_moves():
    from src.directional_model import _dataset
    rows = []
    for symbol in ["A", "B"]:
        for i in range(30):
            rows.append({
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "symbol": symbol,
                "open": 100 + i, "high": 101 + i, "low": 99 + i,
                "close": 100 + i, "volume": 1000,
            })
    hist = pd.DataFrame(rows)
    out = _dataset(hist)
    assert "target_direction" in out.columns
    assert len(out) < len(hist)
    assert (out["target_return"].abs() >= 0.0015).all()
