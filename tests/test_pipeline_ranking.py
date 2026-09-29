import pandas as pd
import pytest

from src.pipeline import fundamental_score


def test_missing_fundamentals_shrink_toward_neutral():
    data = pd.DataFrame({
        "returnOnEquity": [0.10, 0.20],
        "profitMargins": [None, None],
        "revenueGrowth": [None, None],
        "earningsGrowth": [None, None],
        "trailingPE": [None, None],
        "priceToBook": [None, None],
        "debtToEquity": [None, None],
        "dividendYield": [None, None],
    })

    scores = fundamental_score(data)

    assert scores.iloc[0] == pytest.approx(8.4)
    assert scores.iloc[1] == pytest.approx(8.8)
    assert (scores >= 0).all() and (scores <= 20).all()


def test_no_usable_fundamentals_is_neutral():
    data = pd.DataFrame({
        "returnOnEquity": [None, None],
        "trailingPE": [-5.0, 0.0],
    })

    scores = fundamental_score(data)

    assert scores.tolist() == pytest.approx([10.0, 10.0])


def test_multi_horizon_uses_trading_sessions_and_rank_groups():
    from src.ranking_diagnostics import run_multi_horizon_validation

    dates = pd.bdate_range("2026-01-01", periods=25)
    rows = []
    for symbol in ["AAA", "BBB", "CCC", "DDD"]:
        for i, date in enumerate(dates):
            rows.append({"date": date, "symbol": symbol, "close": 100.0 + i})
    history = pd.DataFrame(rows)
    candidates = pd.DataFrame({
        "prediction_date": [dates[0]] * 4,
        "symbol": ["AAA", "BBB", "CCC", "DDD"],
        "rank": [1, 2, 11, 12],
    })

    out = run_multi_horizon_validation(candidates, history)
    top5 = out[(out["group"] == "TOP5") & (out["horizon_sessions"] == 5)].iloc[0]
    next10 = out[(out["group"] == "11-20") & (out["horizon_sessions"] == 5)].iloc[0]

    assert top5["mean_return_pct"] == pytest.approx(5.0)
    assert next10["mean_return_pct"] == pytest.approx(5.0)
    assert top5["symbols"] == 2
    assert top5["lift_vs_universe_pct"] == pytest.approx(0.0)


def test_mfe_mae_uses_only_next_trading_sessions():
    from src.ranking_diagnostics import run_mfe_mae_validation

    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = []
    highs = [101, 110, 103, 104, 105, 106]
    lows = [99, 95, 98, 97, 96, 99]
    closes = [100, 102, 103, 104, 105, 106]
    for symbol in ["AAA", "BBB"]:
        for i, date in enumerate(dates):
            rows.append({
                "date": date, "symbol": symbol,
                "open": closes[i], "high": highs[i], "low": lows[i], "close": closes[i],
            })
    history = pd.DataFrame(rows)
    candidates = pd.DataFrame({
        "prediction_date": [dates[0], dates[0]],
        "symbol": ["AAA", "BBB"],
        "rank": [1, 11],
    })

    out = run_mfe_mae_validation(candidates, history)
    top5 = out[(out["group"] == "TOP5") & (out["horizon_sessions"] == 3)].iloc[0]

    assert top5["mean_mfe_pct"] == pytest.approx(4.0)
    assert top5["mean_mae_pct"] == pytest.approx(-5.0)
    assert top5["mean_final_return_pct"] == pytest.approx(4.0)


def test_mfe_mae_uses_only_next_trading_sessions():
    from src.ranking_diagnostics import run_mfe_mae_validation

    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = []
    highs = [101, 110, 103, 104, 105, 106]
    lows = [99, 95, 98, 97, 96, 99]
    closes = [100, 102, 103, 104, 105, 106]
    for symbol in ["AAA", "BBB"]:
        for i, date in enumerate(dates):
            rows.append({
                "date": date, "symbol": symbol,
                "open": closes[i], "high": highs[i], "low": lows[i], "close": closes[i],
            })
    history = pd.DataFrame(rows)
    candidates = pd.DataFrame({
        "prediction_date": [dates[0], dates[0]],
        "symbol": ["AAA", "BBB"],
        "rank": [1, 11],
    })

    out = run_mfe_mae_validation(candidates, history)
    top5 = out[(out["group"] == "TOP5") & (out["horizon_sessions"] == 3)].iloc[0]

    assert top5["mean_mfe_pct"] == pytest.approx(4.0)
    assert top5["mean_mae_pct"] == pytest.approx(-5.0)
    assert top5["mean_final_return_pct"] == pytest.approx(4.0)
