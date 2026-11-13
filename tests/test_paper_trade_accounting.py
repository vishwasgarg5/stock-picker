import numpy as np
import pandas as pd
import pytest

from src.pipeline import _normalize_paper_trade_accounting


def test_normalizes_legacy_and_fixed_share_rows_without_overwriting_history():
    trades = pd.DataFrame([
        {
            "signal": "BUY", "shares": 10.0, "allocated_capital": 1000.0,
            "entry_price": 100.0, "exit_price": 110.0,
            "gross_profit_loss": 100.0, "costs": 1.0,
            "profit_loss": 99.0, "return_pct": 9.9,
        },
        {
            "signal": "BUY", "quantity": 5, "planned_capital": 500.0,
            "entry_price": 100.0, "exit_price": 110.0,
        },
        {"signal": "SKIP"},
    ])

    result = _normalize_paper_trade_accounting(trades)

    assert result.loc[0, "shares"] == pytest.approx(10.0)
    assert result.loc[0, "costs"] == pytest.approx(1.0)
    assert result.loc[0, "profit_loss"] == pytest.approx(99.0)
    assert result.loc[1, "shares"] == pytest.approx(5.0)
    assert result.loc[1, "allocated_capital"] == pytest.approx(500.0)
    assert result.loc[1, "gross_profit_loss"] == pytest.approx(50.0)
    assert result.loc[1, "costs"] == pytest.approx(0.5)
    assert result.loc[1, "profit_loss"] == pytest.approx(49.5)
    assert result.loc[1, "return_pct"] == pytest.approx(9.9)
    assert result.loc[2, "shares"] == 0.0
    assert result.loc[2, "costs"] == 0.0
    assert result.loc[2, "profit_loss"] == 0.0


def test_does_not_invent_missing_buy_execution_values():
    trades = pd.DataFrame([{
        "signal": "BUY", "quantity": 4, "planned_capital": 400.0,
        "entry_price": 100.0, "exit_price": np.nan,
    }])

    result = _normalize_paper_trade_accounting(trades)

    assert result.loc[0, "shares"] == pytest.approx(4.0)
    assert result.loc[0, "allocated_capital"] == pytest.approx(400.0)
    assert result.loc[0, "costs"] == pytest.approx(0.4)
    assert pd.isna(result.loc[0, "gross_profit_loss"])
    assert pd.isna(result.loc[0, "profit_loss"])
    assert pd.isna(result.loc[0, "return_pct"])
