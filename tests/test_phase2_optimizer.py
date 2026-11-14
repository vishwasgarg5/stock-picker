import numpy as np
import pandas as pd

from src.phase2_optimizer import (
    _attach_repeat_loss_penalty,
    _production_gate,
    _repeat_loss_penalty,
    _shadow_selection_metrics,
)


def test_phase2_gate_requires_both_evidence_thresholds():
    ok, reason = _production_gate({
        "matched_sessions": 17, "v2_trades": 50,
        "direction_lift_pct": 1.0, "return_lift_pct": 0.10,
    })
    assert ok
    assert reason == "all_phase2_gates_passed"


def test_phase2_gate_holds_when_directional_lift_is_weak():
    ok, reason = _production_gate({
        "matched_sessions": 17, "v2_trades": 50,
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


def test_repeat_penalty_replaces_existing_candidate_column_without_suffixes():
    candidates = pd.DataFrame({
        "symbol": ["ABC", "XYZ"],
        "repeat_loss_penalty": [99.0, 99.0],
    })
    repeat = pd.DataFrame({
        "symbol": ["ABC", "XYZ"],
        "repeat_loss_penalty": [1.25, 0.0],
    })

    out = _attach_repeat_loss_penalty(candidates, repeat)

    assert "repeat_loss_penalty_x" not in out.columns
    assert "repeat_loss_penalty_y" not in out.columns
    assert out["repeat_loss_penalty"].tolist() == [1.25, 0.0]


def test_phase2_gate_still_holds_below_17_matched_sessions():
    ok, reason = _production_gate({
        "matched_sessions": 16, "v2_trades": 50,
        "direction_lift_pct": 1.0, "return_lift_pct": 0.10,
    })
    assert not ok
    assert reason == "insufficient_evidence"


def test_phase2_clears_stale_output_when_candidate_input_is_empty(tmp_path, monkeypatch):
    from src import phase2_optimizer as phase2

    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(columns=["prediction_date", "target_date", "symbol"]).to_csv(candidates, index=False)
    output = tmp_path / "phase2.csv"
    pd.DataFrame([{"prediction_date": "2026-01-01", "target_date": "2026-01-02",
                   "symbol": "STALE", "phase2_selected": 1, "phase2_rank": 1}]).to_csv(output, index=False)

    monkeypatch.setattr(phase2, "CANDIDATES", candidates)
    monkeypatch.setattr(phase2, "PREDICTIONS", tmp_path / "missing-predictions.csv")
    monkeypatch.setattr(phase2, "EVALS", tmp_path / "missing-evaluations.csv")
    monkeypatch.setattr(phase2, "HISTORY", tmp_path / "missing-ohlcv.csv")
    monkeypatch.setattr(phase2, "OUT", output)
    monkeypatch.setattr(phase2, "SUMMARY", tmp_path / "summary.csv")
    monkeypatch.setattr(phase2, "STATE", tmp_path / "state.json")

    phase2.run_phase2()

    refreshed = pd.read_csv(output)
    assert refreshed.empty
    assert {"prediction_date", "target_date", "symbol", "phase2_selected", "phase2_rank"}.issubset(
        refreshed.columns
    )


def test_phase2_clears_stale_output_when_all_candidate_dates_are_invalid(tmp_path, monkeypatch):
    from src import phase2_optimizer as phase2

    candidates = tmp_path / "candidates.csv"
    pd.DataFrame({
        "prediction_date": ["not-a-date"],
        "target_date": ["2026-01-02"],
        "symbol": ["ABC"],
    }).to_csv(candidates, index=False)
    output = tmp_path / "phase2.csv"
    pd.DataFrame([{"prediction_date": "2026-01-01", "target_date": "2026-01-02",
                   "symbol": "STALE", "phase2_selected": 1, "phase2_rank": 1}]).to_csv(output, index=False)

    monkeypatch.setattr(phase2, "CANDIDATES", candidates)
    monkeypatch.setattr(phase2, "PREDICTIONS", tmp_path / "missing-predictions.csv")
    monkeypatch.setattr(phase2, "EVALS", tmp_path / "missing-evaluations.csv")
    monkeypatch.setattr(phase2, "HISTORY", tmp_path / "missing-ohlcv.csv")
    monkeypatch.setattr(phase2, "OUT", output)
    monkeypatch.setattr(phase2, "SUMMARY", tmp_path / "summary.csv")
    monkeypatch.setattr(phase2, "STATE", tmp_path / "state.json")

    phase2.run_phase2()

    refreshed = pd.read_csv(output)
    assert refreshed.empty
    assert {"prediction_date", "target_date", "symbol", "phase2_selected", "phase2_rank"}.issubset(
        refreshed.columns
    )

def test_phase2_persists_all_candidate_sessions_without_stale_rows(tmp_path, monkeypatch):
    from src import phase2_optimizer as phase2

    candidates = tmp_path / "candidates.csv"
    pd.DataFrame({
        "prediction_date": ["2026-11-12", "2026-11-13"],
        "target_date": ["2026-11-13", "2026-11-16"],
        "symbol": ["ABC", "XYZ"],
        "score": [80.0, 90.0],
        "rank": [1, 1],
        "prediction_spread": [0.01, 0.02],
        "predicted_close": [101.0, 202.0],
        "base_close": [100.0, 200.0],
    }).to_csv(candidates, index=False)
    output = tmp_path / "phase2.csv"
    pd.DataFrame([{
        "prediction_date": "2026-01-01", "target_date": "2026-01-02",
        "symbol": "STALE", "phase2_selected": 1, "phase2_rank": 1,
    }]).to_csv(output, index=False)

    monkeypatch.setattr(phase2, "CANDIDATES", candidates)
    monkeypatch.setattr(phase2, "PREDICTIONS", tmp_path / "missing-predictions.csv")
    monkeypatch.setattr(phase2, "EVALS", tmp_path / "missing-evaluations.csv")
    monkeypatch.setattr(phase2, "HISTORY", tmp_path / "missing-ohlcv.csv")
    monkeypatch.setattr(phase2, "OUT", output)
    monkeypatch.setattr(phase2, "SUMMARY", tmp_path / "summary.csv")
    monkeypatch.setattr(phase2, "STATE", tmp_path / "state.json")

    phase2.run_phase2()

    refreshed = pd.read_csv(output)
    assert len(refreshed) == 2
    assert set(refreshed["symbol"]) == {"ABC", "XYZ"}
    assert set(pd.to_datetime(refreshed["prediction_date"]).dt.strftime("%Y-%m-%d")) == {
        "2026-11-12", "2026-11-13",
    }

def test_phase2_reconciles_canonical_predictions_missing_from_candidate_history(tmp_path, monkeypatch):
    from src import phase2_optimizer as phase2

    candidates = tmp_path / "candidates.csv"
    pd.DataFrame({
        "prediction_date": ["2026-11-12"],
        "target_date": ["2026-11-13"],
        "symbol": ["OLD"],
        "score": [80.0],
        "rank": [1],
        "prediction_spread": [0.01],
        "predicted_close": [101.0],
        "base_close": [100.0],
    }).to_csv(candidates, index=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({
        "prediction_date": ["2026-11-13"],
        "target_date": ["2026-11-16"],
        "symbol": ["CURRENT"],
        "score": [90.0],
        "rank": [1],
        "prediction_spread": [0.02],
        "predicted_close": [202.0],
        "base_close": [200.0],
    }).to_csv(predictions, index=False)

    monkeypatch.setattr(phase2, "CANDIDATES", candidates)
    monkeypatch.setattr(phase2, "PREDICTIONS", predictions)
    monkeypatch.setattr(phase2, "EVALS", tmp_path / "missing-evaluations.csv")
    monkeypatch.setattr(phase2, "HISTORY", tmp_path / "missing-ohlcv.csv")
    output = tmp_path / "phase2.csv"
    monkeypatch.setattr(phase2, "OUT", output)
    monkeypatch.setattr(phase2, "SUMMARY", tmp_path / "summary.csv")
    monkeypatch.setattr(phase2, "STATE", tmp_path / "state.json")

    phase2.run_phase2()

    refreshed = pd.read_csv(output)
    assert set(refreshed["symbol"]) == {"OLD", "CURRENT"}
    current = refreshed.loc[refreshed["symbol"].eq("CURRENT")].iloc[0]
    assert current["phase2_selected"] == 1
    assert current["phase2_rank"] == 1
