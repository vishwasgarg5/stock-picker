import pandas as pd
import pytest

from src.index_feature_validation import chronological_date_split, _metrics


def test_chronological_split_never_interleaves_future_sessions():
    dates = pd.bdate_range("2025-01-01", periods=100)
    frame = pd.DataFrame({"date": list(dates) * 2})
    train_end, test_start = chronological_date_split(frame, 0.8)
    assert train_end < test_start
    assert train_end == dates[79]
    assert test_start == dates[80]


def test_index_validation_metrics_measure_direction_and_daily_rank():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"] * 5 + ["2026-01-02"] * 5),
        "actual": [0.05, 0.04, 0.03, -0.01, -0.02] * 2,
        "predicted": [0.05, 0.04, 0.03, -0.01, -0.02] * 2,
    })
    result = _metrics(frame)
    assert result["rows"] == 10
    assert result["sessions"] == 2
    assert result["mae_return_pct"] == pytest.approx(0.0)
    assert result["directional_accuracy_pct"] == pytest.approx(100.0)
    assert result["mean_daily_rank_ic"] == pytest.approx(1.0)
    assert result["top_minus_bottom_decile_pct"] > 0


def test_index_validation_rejects_too_few_unique_sessions():
    with pytest.raises(ValueError, match="30 unique sessions"):
        chronological_date_split(pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10)}))


def test_chronological_split_purges_next_session_labels_at_boundary():
    from src.index_feature_validation import chronological_train_test_split
    dates = pd.bdate_range("2026-01-01", periods=40)
    frame = pd.DataFrame({
        "date": list(dates[:-1]) * 2,
        "target_end_date": list(dates[1:]) * 2,
        "symbol": ["AAA"] * 39 + ["BBB"] * 39,
    })
    train, test, train_end, test_start = chronological_train_test_split(frame, 0.80)
    assert train_end < test_start
    assert (train["date"] < test_start).all()
    assert (train["target_end_date"] < test_start).all()
    assert (test["date"] >= test_start).all()
    assert not ((train["date"] == train_end) & (train["target_end_date"] == test_start)).any()


def test_chronological_split_requires_target_endpoint():
    from src.index_feature_validation import chronological_train_test_split
    with pytest.raises(ValueError, match="target_end_date"):
        chronological_train_test_split(pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40)}))


def test_test_sample_cap_keeps_complete_cross_sections():
    from src.index_feature_validation import cap_test_rows_by_complete_sessions
    dates = pd.bdate_range("2026-01-01", periods=5)
    frame = pd.DataFrame([
        {"date": day, "symbol": symbol}
        for day in dates for symbol in ["AAA", "BBB", "CCC", "DDD"]
    ])
    capped = cap_test_rows_by_complete_sessions(frame, 8)
    assert capped["date"].nunique() == 2
    assert len(capped) == 8
    assert capped.groupby("date")["symbol"].nunique().eq(4).all()
    assert capped["date"].min() == dates[-2]
