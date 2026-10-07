import pandas as pd

from src.phase2_optimizer import _production_gate, _repeat_loss_penalty


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
