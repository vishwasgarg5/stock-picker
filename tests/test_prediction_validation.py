import pandas as pd
import pytest

from src.pipeline import _validate_prediction_session


TARGET = pd.Timestamp("2026-11-16")


def valid_predictions():
    rows = []
    for rank in range(1, 11):
        rows.append({
            "symbol": f"STK{rank}",
            "target_date": TARGET,
            "rank": rank,
            "base_close": 100.0,
            "predicted_open": 100.5,
            "predicted_high": 103.0,
            "predicted_low": 98.0,
            "predicted_close": 101.0,
        })
    return pd.DataFrame(rows)


def test_valid_prediction_session_passes():
    _validate_prediction_session(valid_predictions(), TARGET)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("symbol", "   ", "blank stock symbol"),
        ("symbol", " stk2 ", "duplicate symbols"),
        ("rank", 1.5, "ranks must be"),
        ("rank", 11, "ranks must be"),
        ("rank", float("inf"), "ranks must be"),
        ("base_close", 0.0, "non-finite or non-positive"),
        ("predicted_close", float("nan"), "non-finite or non-positive"),
        ("predicted_open", 150.0, "predicted move exceeds"),
        ("predicted_high", 99.0, "predicted high is below"),
        ("predicted_low", 102.0, "predicted low is above"),
    ],
)
def test_prediction_integrity_gate_rejects_invalid_rows(column, value, message):
    prediction = valid_predictions()
    prediction.loc[0, column] = value

    with pytest.raises(RuntimeError, match=message):
        _validate_prediction_session(prediction, TARGET)


def test_prediction_integrity_gate_rejects_missing_required_columns():
    prediction = valid_predictions().drop(columns=["predicted_low"])

    with pytest.raises(RuntimeError, match="missing columns"):
        _validate_prediction_session(prediction, TARGET)


def test_prediction_integrity_gate_rejects_target_date_mismatch():
    prediction = valid_predictions()
    prediction.loc[0, "target_date"] = pd.Timestamp("2026-11-17")

    with pytest.raises(RuntimeError, match="target date mismatch"):
        _validate_prediction_session(prediction, TARGET)
