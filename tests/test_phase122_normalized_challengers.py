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
    # Make BBB an exact price-scale copy of AAA.
    history.loc[history["symbol"] == "BBB", ["open", "high", "low", "close"]] *= 10.0
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
