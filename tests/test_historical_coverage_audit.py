import pandas as pd

from src.historical_coverage_audit import audit_history


def test_audit_reports_no_history_without_inventing_rows():
    universe = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    summary, detail = audit_history(universe, pd.DataFrame())

    assert summary["status"] == "BLOCKED_NO_HISTORY"
    assert summary["symbols_without_history"] == 2
    assert summary["total_valid_rows"] == 0
    assert detail["observed_sessions"].sum() == 0
    assert set(detail["status"]) == {"NO_HISTORY"}


def test_audit_reports_per_symbol_depth_and_missing_sessions():
    dates = pd.bdate_range("2024-01-01", periods=65)
    rows = []
    for symbol, selected in [("AAA", dates), ("BBB", dates[:10])]:
        for date in selected:
            rows.append({
                "date": date, "symbol": symbol, "open": 10.0,
                "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000,
            })
    summary, detail = audit_history(
        pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "DUMMY_X"]}),
        pd.DataFrame(rows),
    )

    by_symbol = detail.set_index("symbol")
    assert summary["universe_symbols"] == 3
    assert by_symbol.loc["AAA", "observed_sessions"] == 65
    assert by_symbol.loc["AAA", "training_ready_50_sessions"]
    assert by_symbol.loc["BBB", "observed_sessions"] == 10
    assert not by_symbol.loc["BBB", "training_ready_50_sessions"]
    assert by_symbol.loc["CCC", "status"] == "NO_HISTORY"
    assert by_symbol.loc["AAA", "missing_sessions"] >= 0


def test_listing_date_sets_each_stock_history_start():
    universe = pd.DataFrame({"symbol": ["NEWCO"]})
    dates = pd.bdate_range("2024-01-01", periods=65)
    history = pd.DataFrame([{"date": day, "symbol": "NEWCO", "open": 10.0, "high": 11.0,
        "low": 9.0, "close": 10.5, "volume": 1000} for day in dates])
    listing = pd.DataFrame({"symbol": ["NEWCO"], "listing_date": ["2024-01-01"]})
    _, detail = audit_history(universe, history, listing)
    assert detail.iloc[0]["listing_date"] == "2024-01-01"
    assert detail.iloc[0]["history_starts_at_first_possible_session"]
    assert detail.iloc[0]["listing_history_status"] == "STARTS_AT_FIRST_POSSIBLE_SESSION"


def test_listing_audit_flags_missing_history_at_the_beginning():
    universe = pd.DataFrame({"symbol": ["NEWCO"]})
    dates = pd.bdate_range("2024-01-15", periods=60)
    history = pd.DataFrame([{"date": day, "symbol": "NEWCO", "open": 10.0, "high": 11.0,
        "low": 9.0, "close": 10.5, "volume": 1000} for day in dates])
    listing = pd.DataFrame({"symbol": ["NEWCO"], "listing_date": ["2024-01-01"]})
    _, detail = audit_history(universe, history, listing)
    assert detail.iloc[0]["listing_to_first_bar_gap_sessions"] > 0
    assert detail.iloc[0]["listing_history_status"] == "STARTS_AFTER_LISTING_DATE_REVIEW"


def test_listing_date_is_not_inferred_when_missing():
    universe = pd.DataFrame({"symbol": ["NEWCO"]})
    dates = pd.bdate_range("2024-01-01", periods=60)
    history = pd.DataFrame([{"date": day, "symbol": "NEWCO", "open": 10.0, "high": 11.0,
        "low": 9.0, "close": 10.5, "volume": 1000} for day in dates])
    _, detail = audit_history(universe, history)
    assert pd.isna(detail.iloc[0]["listing_date"])
    assert detail.iloc[0]["listing_history_status"] == "LISTING_DATE_UNKNOWN"


def test_official_nse_quote_listing_date_parser():
    from src.listing_master import normalise_quote
    row = normalise_quote("ABC", {"info": {"listingDate": "15-Jan-2024",
        "isin": "INE123A01016", "companyName": "ABC Limited"}})
    assert row["listing_date"] == "2024-01-15"
    assert row["isin"] == "INE123A01016"
