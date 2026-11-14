import numpy as np
import pandas as pd
import pytest

from src.paper_trading_v2 import _net_return_pct, _phase2_selection_flags


def test_v2_return_pct_uses_net_pnl_and_actual_entry_notional():
    result = _net_return_pct(
        pd.Series([98.95, -51.05, 0.0]),
        pd.Series([100.0, 100.0, 100.0]),
        pd.Series([10, 10, 0]),
    )

    assert result.iloc[0] == pytest.approx(9.895)
    assert result.iloc[1] == pytest.approx(-5.105)
    assert np.isnan(result.iloc[2])


def test_missing_phase2_selection_metadata_fails_closed():
    predictions = pd.DataFrame({"symbol": ["AAA", "BBB"]})

    selected = _phase2_selection_flags(predictions)

    assert selected.tolist() == [False, False]
    assert selected.index.equals(predictions.index)


def test_malformed_phase2_selection_metadata_fails_closed():
    predictions = pd.DataFrame({"phase2_selected": [1, 0, None, "invalid", 2]})

    selected = _phase2_selection_flags(predictions)

    assert selected.tolist() == [True, False, False, False, False]
