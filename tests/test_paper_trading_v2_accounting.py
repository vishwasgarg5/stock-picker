import numpy as np
import pandas as pd
import pytest

from src.paper_trading_v2 import _net_return_pct


def test_v2_return_pct_uses_net_pnl_and_actual_entry_notional():
    result = _net_return_pct(
        pd.Series([98.95, -51.05, 0.0]),
        pd.Series([100.0, 100.0, 100.0]),
        pd.Series([10, 10, 0]),
    )

    assert result.iloc[0] == pytest.approx(9.895)
    assert result.iloc[1] == pytest.approx(-5.105)
    assert np.isnan(result.iloc[2])
