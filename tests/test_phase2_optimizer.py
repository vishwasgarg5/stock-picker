import numpy as np
import pandas as pd

from src.phase2_optimizer import (
    _production_gate,
    _repeat_loss_penalty,
    _shadow_selection_metrics,
)


def test_phase2_gate_requires_both_evidence_thresholds():
    ok, reason = _production_gate({
        "matched_sessions": 20, "v2_trades": 50,
        "direction_lift_pct": 1.0, "return_lift_pct": 0.10,
    })
    assert ok
    assert reason == "all_phase2_gates_passed"


def test_phase2_gate_holds_when_directional_lift_is_weak():
    ok, reason = _production_gate({
        "matched_sessions": 20, "v2_trades": 50,
        "direction_lift_pct": 0.1, "return_lift_pct": 1.0,
    })
    assert not ok
    assert reason == "directional_lift_below_gate"


def test_repeat_loss_penalty_is_bounded():
    c = pd.DataFrame({"symbol": ["ABC", "XYZ"]})
    e = pd.DataFrame({
        "symbol": ["ABC"] * 8 + ["XYZ"],
        "target_date": pd.date_range("2026-09-01", periods=9),
        "profit_loss": [-100.0] * 8 + [100.0],
    })
    out = _repeat_loss_penalty(c, e)
    assert out["repeat_loss_penalty"].max() <= 2.0
    assert float(out.loc[out["symbol"] == "ABC", "repeat_loss_penalty"].iloc[0]) > float(
        out.loc[out["symbol"] == "XYZ", "repeat_loss_penalty"].iloc[0]
    )


def test_shadow_metrics_measure_direction_and_return_lift():
    c = pd.DataFrame({
        "prediction_date": pd.to_datetime(["2026-10-01"] * 4),
        "symbol": ["A", "B", "C", "D"],
        "phase2_selected": [1, 1, 0, 0],
        "phase2_rank": [1, 2, 11, 12],
        "rank": [4, 5, 1, 2],
    })
    e = pd.DataFrame({
        "prediction_date": pd.to_datetime(["2026-10-01"] * 4),
        "symbol": ["A", "B", "C", "D"],
        "rank": [4, 5, 1, 2],
        "actual_open": [100, 100, 100, 100],
        "actual_close": [103, 102, 99, 98],
        "close_direction_correct": [1, 1, 0, 0],
    })
    direction, ret, sessions = _shadow_selection_metrics(c, e)
    assert sessions == 1
    assert direction > 0
    assert ret > 0


def test_shadow_metrics_are_safe_with_missing_values():
    c = pd.DataFrame({
        "prediction_date": pd.to_datetime(["2026-10-01"]),
        "symbol": ["A"],
        "phase2_selected": [1],
        "phase2_rank": [1],
        "rank": [1],
    })
    e = pd.DataFrame({
        "prediction_date": pd.to_datetime(["2026-10-01"]),
        "symbol": ["A"],
        "rank": [1],
        "actual_open": [np.nan],
        "actual_close": [100.0],
        "close_direction_correct": [np.nan],
    })
    direction, ret, sessions = _shadow_selection_metrics(c, e)
    assert sessions == 1
    assert direction == 0.0
    assert ret == 0.0
