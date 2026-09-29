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
