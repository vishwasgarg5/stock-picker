import numpy as np
import pandas as pd
import pytest
from src.phase125_feature_ablation import run_ablation, ABLATIONS
from src.pipeline import FEATURE_COLUMNS

def _frame():
    rng = np.random.default_rng(125)
    dates = pd.bdate_range("2024-01-01", periods=65)
    rows = []
    for i, date in enumerate(dates[:-1]):
        for symbol in range(24):
            latent = rng.normal()
            row = {"date": date, "target_date": dates[i+1], "symbol": f"S{symbol:02d}",
                   "target_close_return": 0.004 * latent + 0.02 * rng.normal()}
            for feature in FEATURE_COLUMNS:
                row[feature] = latent + rng.normal(scale=0.8)
            rows.append(row)
    return pd.DataFrame(rows)

def test_ablation_uses_strict_chronological_label_cutoff():
    summary, detail = run_ablation(_frame(), n_folds=4)
    assert summary["folds_completed"] == 4
    assert summary["strict_no_lookahead"] is True
    assert detail["strict_label_cutoff"].all()
    for row in detail.to_dict("records"):
        assert row["train_end_target_date"] < row["test_start_date"]

def test_ablation_variants_are_predeclared_and_keep_control():
    assert "incumbent_all_features" in ABLATIONS
    assert "without_price_level_indicators" in ABLATIONS
    assert "compact_returns_risk_volume" in ABLATIONS
    assert set(ABLATIONS["incumbent_all_features"]) == set(FEATURE_COLUMNS)

def test_duplicate_date_symbol_keys_fail_closed():
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        run_ablation(frame, n_folds=4)

def test_missing_feature_fails_closed():
    with pytest.raises(ValueError, match="missing"):
        run_ablation(_frame().drop(columns=[FEATURE_COLUMNS[0]]), n_folds=4)
