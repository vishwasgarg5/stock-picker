import numpy as np
import pandas as pd

from src import index_constituent_intelligence as ici


def test_index_bars_keep_ohlcv_and_are_normalised():
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    raw = pd.DataFrame({"Open": [100, 101, 102], "High": [102, 103, 104],
                        "Low": [99, 100, 101], "Close": [101, 102, 103],
                        "Volume": [0, 0, 0]}, index=dates)
    out = ici.normalise_index_bars("NIFTY50", "^NSEI", raw)
    assert list(out.columns) == ici.INDEX_COLUMNS
    assert len(out) == 3
    assert out.iloc[-1]["close"] == 103
    assert out.iloc[-1]["volume"] == 0


def test_constituent_weight_is_marked_estimated_when_derived_from_ffmc():
    frame = ici.normalise_constituent_payload("NIFTY50", {"data": [
        {"symbol": "AAA", "ffmc": 600}, {"symbol": "BBB", "ffmc": 400},
        {"symbol": "NIFTY 50", "ffmc": 1000},
    ]}, "2024-01-02")
    assert set(frame.symbol) == {"AAA", "BBB"}
    weights = frame.set_index("symbol").weight_pct
    assert weights["AAA"] == 60
    assert weights["BBB"] == 40
    assert set(frame.weight_method) == {"ESTIMATED_FFMC_PROXY"}


def test_official_constituent_weight_is_preserved():
    frame = ici.normalise_constituent_payload("NIFTY50", {"data": [
        {"symbol": "AAA", "weightage": "12.5", "ffmc": 600},
        {"symbol": "BBB", "weightage": "7.5", "ffmc": 400},
    ]}, "2024-01-02")
    assert frame.weight_pct.tolist() == [12.5, 7.5]
    assert set(frame.weight_method) == {"OFFICIAL_NSE_WEIGHT"}


def test_features_do_not_backfill_constituent_membership_before_snapshot():
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    stock = pd.DataFrame({"date": dates, "symbol": "AAA", "close": np.arange(100, 125)})
    index = pd.DataFrame({"date": dates, "index": "NIFTY50", "close": np.arange(200, 225)})
    membership = pd.DataFrame({"index": ["NIFTY50"], "symbol": ["AAA"],
        "as_of_date": ["2024-01-20"], "weight_pct": [10.0], "weight_method": ["OFFICIAL_NSE_WEIGHT"]})
    out = ici.build_constituent_features(stock, index, membership)
    assert not out.empty
    assert out.date.min() >= pd.Timestamp("2024-01-20")
    assert out.weight_pct.eq(10.0).all()
    assert out.as_of_date.eq(pd.Timestamp("2024-01-20")).all()


def test_features_compute_stock_index_relationships():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    index_returns = np.linspace(-0.01, 0.02, len(dates))
    index_close = 100 * np.cumprod(1 + index_returns)
    stock_close = 50 * np.cumprod(1 + 1.5 * index_returns)
    stock = pd.DataFrame({"date": dates, "symbol": "AAA", "close": stock_close})
    index = pd.DataFrame({"date": dates, "index": "NIFTY50", "close": index_close})
    membership = pd.DataFrame({"index": ["NIFTY50"], "symbol": ["AAA"],
        "as_of_date": ["2024-01-01"], "weight_pct": [5.0], "weight_method": ["OFFICIAL_NSE_WEIGHT"]})
    out = ici.build_constituent_features(stock, index, membership)
    last = out.iloc[-1]
    assert np.isfinite(last.rolling_corr_60d)
    assert last.rolling_beta_60d > 0
    assert last.weighted_stock_return_contribution_pct != 0


def test_estimated_weights_only_fill_residual_after_official_weights():
    frame = ici.normalise_constituent_payload("NIFTY50", {"data": [
        {"symbol": "AAA", "weightage": "20", "ffmc": 600},
        {"symbol": "BBB", "ffmc": 300},
        {"symbol": "CCC", "ffmc": 100},
    ]}, "2024-01-02")
    weights = frame.set_index("symbol").weight_pct
    assert weights["AAA"] == 20
    assert weights["BBB"] == 60
    assert weights["CCC"] == 20
    assert weights.sum() == 100


def test_constituent_snapshot_preserves_exact_source_endpoint():
    source = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    frame = ici.normalise_constituent_payload("NIFTY50", {
        "data": [{"symbol": "AAA", "weightage": "10"}],
    }, "2024-01-02", source_url=source)
    assert frame.loc[0, "source"] == source
