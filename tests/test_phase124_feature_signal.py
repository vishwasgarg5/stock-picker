import numpy as np
import pandas as pd
import pytest
from src.phase124_feature_signal import analyse_feature_signal

def _frame():
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for i, date in enumerate(dates[:-1]):
        for symbol in range(20):
            signal = rng.normal()
            rows.append({"date": date, "target_date": dates[i+1], "symbol": f"S{symbol:02d}",
                         "signal": signal, "noise": rng.normal(),
                         "target_close_return": 0.02 * signal + 0.03 * rng.normal()})
    return pd.DataFrame(rows)

def test_expanding_folds_have_strict_label_cutoff_and_oos_signal():
    summary, folds = analyse_feature_signal(_frame(), ["signal", "noise"], n_folds=4)
    assert summary["strict_no_lookahead"] is True
    assert summary["folds_completed"] == 4
    assert folds["strict_label_cutoff"].all()
    signal = folds.loc[folds.feature == "signal", "test_mean_rank_ic"].dropna()
    noise = folds.loc[folds.feature == "noise", "test_mean_rank_ic"].dropna()
    assert signal.mean() > noise.mean()

def test_duplicate_date_symbol_keys_fail_closed():
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        analyse_feature_signal(frame, ["signal"], n_folds=4)

def test_training_labels_must_precede_test_feature_date():
    _, folds = analyse_feature_signal(_frame(), ["signal"], n_folds=4)
    for row in folds.to_dict("records"):
        assert row["train_end_target_date"] < row["test_start_date"]

def test_missing_required_feature_fails_closed():
    with pytest.raises(ValueError, match="missing"):
        analyse_feature_signal(_frame().drop(columns=["signal"]), ["signal"], n_folds=4)
