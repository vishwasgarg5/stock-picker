import numpy as np
import pandas as pd
import pytest

from src.phase123_magnitude_calibration import evaluate_magnitude_calibration


def _scores():
    dates = pd.bdate_range("2025-01-01", periods=16)
    rows = []
    for i, date in enumerate(dates):
        target = dates[i + 1]
        for j in range(4):
            base = 100.0 + j * 20
            actual_return = 0.01 * (1 if (i + j) % 2 else -1)
            # Forecasts are deliberately compressed to one quarter of actual.
            predicted_return = actual_return * 0.25
            rows.append({
                "prediction_date": date, "target_date": target,
                "symbol": f"S{j}", "base_close": base,
                "predicted_close": base * (1 + predicted_return),
                "actual_close": base * (1 + actual_return),
            })
    return pd.DataFrame(rows)


def test_calibration_uses_only_fully_matured_prior_labels_and_recovers_scale():
    daily, report = evaluate_magnitude_calibration(
        _scores(), min_train_rows=8, min_train_sessions=2, max_scale=5
    )
    eligible = daily.loc[daily["calibration_applied"]]
    assert not eligible.empty
    assert eligible["no_lookahead"].all()
    assert (pd.to_datetime(eligible["latest_training_target_date"]) <
            pd.to_datetime(eligible["prediction_date"])).all()
    assert np.allclose(eligible["scale_alpha"], 4.0)
    assert report["production_model_changed"] is False
    assert report["automatic_promotion"] is False


def test_duplicate_forecast_keys_fail_closed():
    scores = _scores()
    scores = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate prediction_date"):
        evaluate_magnitude_calibration(scores)


def test_missing_training_history_falls_back_to_identity():
    daily, _ = evaluate_magnitude_calibration(
        _scores().iloc[:4], min_train_rows=30, min_train_sessions=5
    )
    assert not daily["calibration_applied"].any()
    assert daily["scale_alpha"].isna().all()


def test_nonpositive_prices_are_excluded_from_valid_cohort():
    scores = _scores()
    scores.loc[0, "actual_close"] = -1
    _, report = evaluate_magnitude_calibration(scores, min_train_rows=8, min_train_sessions=2)
    assert report["scored_rows"] == len(scores) - 1
