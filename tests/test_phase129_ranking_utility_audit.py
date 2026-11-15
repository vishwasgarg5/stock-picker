import numpy as np
import pandas as pd
import pytest

from src.phase129_ranking_utility_audit import evaluate_ranking_utility


def _data(n_dates=50):
    dates = pd.bdate_range("2025-01-01", periods=n_dates + 1)
    candidates, history = [], []
    for j in range(20):
        symbol = f"S{j:02d}"
        base = 100.0 + j
        closes = [base]
        for i in range(n_dates):
            ret = (10 - j) * 0.001 + (i % 4) * 0.0001
            closes.append(closes[-1] * (1 + ret))
        for i, date in enumerate(dates):
            history.append({"date": date, "symbol": symbol, "close": closes[i]})
        for i, date in enumerate(dates[:-1]):
            candidates.append({
                "prediction_date": date, "target_date": dates[i+1], "symbol": symbol,
                "rank": j + 1, "score": 20 - j, "base_close": closes[i],
            })
    return pd.DataFrame(candidates), pd.DataFrame(history)


def test_phase129_measures_rank_ic_spread_turnover_and_costs():
    c, h = _data()
    summary, detail = evaluate_ranking_utility(c, h, n_folds=4, min_review_sessions=17)
    assert summary["prediction_sessions_evaluated"] >= 17
    assert summary["status"] == "READY_FOR_REVIEW"
    assert summary["rank_ic_spearman_mean"] > 0
    assert summary["top_bottom_spread_mean_pct"] > 0
    assert (detail.transaction_cost_pct >= 0).all()
    assert (detail.top5_net_return_pct <= detail.top5_mean_return_pct).all()
    assert summary["production"]["changed"] is False


def test_duplicate_candidate_keys_fail_closed():
    c, h = _data()
    c = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_ranking_utility(c, h)


def test_exact_target_date_required():
    c, h = _data()
    h = h.loc[~((h.date == c.iloc[0].target_date) & (h.symbol == c.iloc[0].symbol))]
    summary, _ = evaluate_ranking_utility(c, h)
    assert summary["excluded_incomplete_sessions"] == 1


def test_review_threshold_is_not_performance_gate():
    c, h = _data()
    summary, _ = evaluate_ranking_utility(c, h, min_review_sessions=100)
    assert summary["status"] == "INSUFFICIENT_SESSIONS"
    assert summary["performance_gate_met"] is False
