import pandas as pd
import pytest

from src import paper_trading as pt


def _trade(symbol, date, entry, exit_price, rank=1):
    return {
        "symbol": symbol,
        "target_date": pd.Timestamp(date),
        "rank": rank,
        "signal": "BUY",
        "entry_price": entry,
        "exit_price": exit_price,
    }


def test_equal_capital_sizing_and_net_pnl():
    trades = pd.DataFrame([
        _trade("AAA", "2026-01-02", 100.0, 110.0, 1),
        _trade("BBB", "2026-01-02", 200.0, 180.0, 2),
    ])

    updated, daily = pt._build_portfolio(trades)

    # ₹50,000 allocated to each stock; 0.10% round-trip cost per trade.
    assert updated["allocated_capital"].tolist() == pytest.approx([50_000.0, 50_000.0])
    assert updated["shares"].tolist() == pytest.approx([500.0, 250.0])
    assert updated["gross_profit_loss"].tolist() == pytest.approx([5_000.0, -5_000.0])
    assert updated["costs"].tolist() == pytest.approx([50.0, 50.0])
    assert updated["profit_loss"].tolist() == pytest.approx([4_950.0, -5_050.0])
    assert updated["return_pct"].tolist() == pytest.approx([9.9, -10.1])

    assert daily.iloc[0]["gross_pnl"] == pytest.approx(0.0)
    assert daily.iloc[0]["costs"] == pytest.approx(100.0)
    assert daily.iloc[0]["net_pnl"] == pytest.approx(-100.0)
    assert daily.iloc[0]["portfolio_value"] == pytest.approx(99_900.0)


def test_portfolio_compounds_from_previous_session():
    trades = pd.DataFrame([
        _trade("AAA", "2026-01-02", 100.0, 110.0, 1),
        _trade("BBB", "2026-01-05", 100.0, 110.0, 1),
    ])

    updated, daily = pt._build_portfolio(trades)

    # First session: ₹1,000 gross, ₹10 cost => ₹100,990? No:
    # one trade receives all ₹100,000, so net = ₹9,900.
    assert daily.iloc[0]["portfolio_value"] == pytest.approx(109_900.0)
    # Second session uses the new portfolio as its allocation base.
    assert updated.iloc[1]["allocated_capital"] == pytest.approx(109_900.0)
    assert daily.iloc[1]["portfolio_value"] == pytest.approx(120_770.1)


def test_both_target_and_stop_uses_conservative_stop():
    row = pd.Series({
        "symbol": "AAA",
        "target_date": pd.Timestamp("2026-01-02"),
        "rank": 1,
        "base_close": 100.0,
        "actual_open": 100.0,
        "actual_high": 110.0,
        "actual_low": 90.0,
        "actual_close": 105.0,
        "predicted_open": 101.0,
        "predicted_high": 106.0,
        "predicted_low": 96.0,
        "predicted_close": 103.0,
        "predicted_return": 0.03,
        "predicted_upside": 0.06,
        "predicted_downside": -0.04,
        "rank": 1,
        "score": 80.0,
    })

    result = pt._simulate_day(row, None, None, None, learned=False)

    assert result["signal"] == "BUY"
    assert result["exit_reason"] == "stop_and_target_same_day_conservative"
    assert result["exit_price"] < row["actual_open"]


def test_training_rows_only_use_matching_actual_session():
    predictions = pd.DataFrame([{
        "symbol": "AAA",
        "prediction_date": "2026-01-01",
        "target_date": "2026-01-02",
        "base_close": 100.0,
        "predicted_open": 101.0,
        "predicted_high": 105.0,
        "predicted_low": 98.0,
        "predicted_close": 103.0,
        "rank": 1,
        "score": 80.0,
    }])
    history = pd.DataFrame([{
        "symbol": "AAA",
        "date": "2026-01-02",
        "open": 102.0,
        "high": 106.0,
        "low": 100.0,
        "close": 104.0,
    }])

    rows = pt._build_training_rows(predictions, history)

    assert len(rows) == 1
    assert rows.iloc[0]["actual_open"] == pytest.approx(102.0)
    assert rows.iloc[0]["close_return"] == pytest.approx(104 / 102 - 1)
