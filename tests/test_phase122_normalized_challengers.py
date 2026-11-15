import numpy as np
import pandas as pd

from src.phase122_normalized_challengers import (
    build_normalized_features,
    run_challenger_experiment,
)


def _history(sessions=150, symbols=("AAA", "BBB", "CCC", "DDD")):
    dates = pd.bdate_range("2024-01-01", periods=sessions)
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        scale = 1.0 + symbol_index * 7.0
        for i, date in enumerate(dates):
            close = scale * (100.0 + 0.08 * i + 2.0 * np.sin(i / 5.0 + symbol_index))
            open_price = close * (1.0 + 0.001 * np.cos(i))
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            rows.append({
                "date": date, "symbol": symbol, "open": open_price,
                "high": high, "low": low, "close": close,
                "volume": 100000.0 + 100.0 * i,
                "stock_splits": 0.0, "dividends": 0.0,
            })
    return pd.DataFrame(rows)


def test_price_level_indicators_are_scale_normalized():
    history = _history(sessions=80, symbols=("AAA", "BBB"))
    # Make BBB an exact price-scale copy of AAA (same path, different units).
    aaa = history.loc[history["symbol"] == "AAA"].sort_values("date")
    bbb_mask = history["symbol"] == "BBB"
    for column in ["open", "high", "low", "close"]:
        history.loc[bbb_mask, column] = aaa[column].to_numpy() * 10.0
    normalized = build_normalized_features(history)
    a = normalized.loc[normalized.symbol == "AAA"].sort_values("date").tail(1)
    b = normalized.loc[normalized.symbol == "BBB"].sort_values("date").tail(1)
    for column in ["sma20_rel_close", "sma50_rel_close", "ema20_rel_close",
                   "ema50_rel_close", "macd_rel_close", "macd_signal_rel_close"]:
        assert np.isclose(a[column].iloc[0], b[column].iloc[0], rtol=1e-8, atol=1e-10)


def test_challengers_use_expanding_folds_and_strict_label_cutoff():
    report = run_challenger_experiment(
        _history(), initial_train_fraction=0.55, n_splits=3,
        min_train_rows=100, ridge_alpha=10.0,
    )
    assert report["status"] == "SHADOW_EXPERIMENT_COMPLETE"
    assert report["production_model_changed"] is False
    assert report["production_champion"] == "V1"
    assert report["challengers_shadow_only"] is True
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["no_lookahead"] is True
        assert pd.Timestamp(fold["latest_training_target_date"]) < pd.Timestamp(fold["test_start"])
        assert set(fold["models"]) == {
            "incumbent_control", "normalized_ensemble",
            "regularized_return", "previous_close_baseline",
        }
    assert set(report["aggregate_metrics"]) == {
        "incumbent_control", "normalized_ensemble",
        "regularized_return", "previous_close_baseline",
    }
    for metrics in report["aggregate_metrics"].values():
        assert metrics["rows"] > 0
        assert metrics["mape_pct"] is not None
        assert metrics["directional_accuracy_pct"] is not None
        assert metrics["signed_bias_pct_points"] is not None
        assert metrics["forecast_to_realized_magnitude_ratio"] is not None


def test_duplicate_history_keys_fail_closed():
    history = _history(sessions=70)
    duplicated = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    try:
        run_challenger_experiment(duplicated, min_train_rows=10)
    except ValueError as exc:
        assert "Duplicate date/symbol" in str(exc)
    else:
        raise AssertionError("duplicate history keys must be rejected")



def test_future_split_does_not_rewrite_earlier_feature_rows():
    history = _history(sessions=90, symbols=("AAA",))
    with_future_split = history.copy()
    split_date = pd.Timestamp(with_future_split["date"].sort_values().unique()[65])
    with_future_split.loc[
        (with_future_split["symbol"] == "AAA") & (with_future_split["date"] == split_date),
        "stock_splits",
    ] = 2.0
    baseline = build_normalized_features(history)
    adjusted = build_normalized_features(with_future_split)
    cutoff = split_date - pd.offsets.BDay(1)
    columns = ["sma20", "sma50", "ema20", "ema50", "macd", "macd_signal",
               "return_1d", "return_5d", "return_20d", "atr14_pct"]
    left = baseline.loc[baseline["date"] <= cutoff].sort_values("date")
    right = adjusted.loc[adjusted["date"] <= cutoff].sort_values("date")
    for column in columns:
        assert np.allclose(left[column], right[column], equal_nan=True)


def test_next_session_split_is_accounted_for_in_realized_return():
    history = _history(sessions=75, symbols=("AAA",))
    split_date = pd.Timestamp(history["date"].sort_values().unique()[60])
    mask = (history["symbol"] == "AAA") & (history["date"] >= split_date)
    history.loc[mask, ["open", "high", "low", "close"]] *= 0.5
    history.loc[
        (history["symbol"] == "AAA") & (history["date"] == split_date),
        "stock_splits",
    ] = 2.0
    features = build_normalized_features(history)
    prior_date = pd.Timestamp(history["date"].sort_values().unique()[59])
    prior = features.loc[features["date"] == prior_date].iloc[0]
    assert abs(prior["target_close_return"]) < 0.01



def test_cash_dividend_is_point_in_time_adjusted_without_future_rewrite():
    history = _history(sessions=90, symbols=("AAA",))
    dates = sorted(pd.to_datetime(history["date"]).unique())
    ex_date = pd.Timestamp(dates[60])
    prior_close = float(history.loc[history["date"] == dates[59], "close"].iloc[0])
    dividend = 2.0
    after = history["date"] >= ex_date
    for column in ["open", "high", "low", "close"]:
        history.loc[after, column] -= dividend
    history.loc[history["date"] == ex_date, "dividends"] = dividend

    no_dividend = _history(sessions=90, symbols=("AAA",))
    baseline = build_normalized_features(no_dividend)
    adjusted = build_normalized_features(history)

    prior_date = pd.Timestamp(dates[59])
    prior = adjusted.loc[adjusted["date"] == prior_date].iloc[0]
    ex_row = adjusted.loc[adjusted["date"] == ex_date].iloc[0]
    baseline_prior = baseline.loc[baseline["date"] == prior_date].iloc[0]
    baseline_ex = baseline.loc[baseline["date"] == ex_date].iloc[0]
    assert np.isclose(prior["target_close_return"], baseline_prior["target_close_return"], atol=1e-10)
    assert np.isclose(ex_row["return_1d"], baseline_ex["return_1d"], atol=1e-10)

    cutoff = ex_date - pd.offsets.BDay(1)
    left = baseline.loc[baseline["date"] <= cutoff].sort_values("date")
    right = adjusted.loc[adjusted["date"] <= cutoff].sort_values("date")
    for column in ["sma20", "ema20", "macd", "macd_signal", "adjusted_close"]:
        assert np.allclose(left[column], right[column], equal_nan=True)
    assert prior_close > dividend
