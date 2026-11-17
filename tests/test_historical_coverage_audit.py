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



def test_listing_master_stops_repeated_systemic_nse_failures(tmp_path, monkeypatch):
    from requests import HTTPError
    from src import listing_master

    class Response:
        def raise_for_status(self):
            raise HTTPError("403 Client Error: Forbidden")

    class Session:
        def __init__(self):
            self.headers = {}
            self.quote_calls = 0

        def get(self, url, timeout):
            if url == listing_master.HOME:
                class Home:
                    pass
                return Home()
            self.quote_calls += 1
            return Response()

    session = Session()
    monkeypatch.setattr(listing_master, "DATA", tmp_path)
    monkeypatch.setattr(listing_master, "FILE", tmp_path / "nse_listing_master.csv")
    result = listing_master.refresh_listing_master(
        [f"SYM{i}" for i in range(30)], session=session,
        delay_seconds=0,
    )
    assert session.quote_calls == 10
    assert result["symbols_fetched"] == 0
    assert len(result["symbols_unresolved"]) == 30
    assert any("Skipped after 10 consecutive systemic NSE errors" in value for value in result["lookup_failures"].values())


def test_nse_security_master_listing_date_parser_uses_named_columns():
    import gzip
    from src.listing_master import parse_security_master
    payload = b"Symbol,Company Name,ISIN,Listing Date\nABC,ABC Limited,INE123A01016,20240115\nOTHER,Other Ltd,INE123A01024,20240201\n"
    found = parse_security_master(gzip.compress(payload), {"ABC"},
        "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_15012024.csv.gz")
    assert len(found) == 1
    assert found[0]["symbol"] == "ABC"
    assert found[0]["listing_date"] == "2024-01-15"
    assert "nsearchives.nseindia.com" in found[0]["listing_date_source"]


def test_listing_master_persists_schema_when_official_sources_unavailable(tmp_path, monkeypatch):
    from requests import HTTPError
    from src import listing_master
    class Response:
        content = b""
        def raise_for_status(self): raise HTTPError("403 Forbidden")
    class Session:
        def __init__(self): self.headers = {}
        def get(self, url, timeout, **kwargs):
            if url == listing_master.HOME: return object()
            return Response()
    monkeypatch.setattr(listing_master, "DATA", tmp_path)
    monkeypatch.setattr(listing_master, "FILE", tmp_path / "nse_listing_master.csv")
    result = listing_master.refresh_listing_master(["ABC"], session=Session(), delay_seconds=0, max_attempts=1)
    assert result["status"] == "SOURCE_UNAVAILABLE"
    assert listing_master.FILE.exists()
    assert list(pd.read_csv(listing_master.FILE).columns) == listing_master.COLS


def test_nse_pipe_security_master_uses_documented_listing_date_field():
    import gzip
    from src.listing_master import parse_security_master
    fields = [""] * 54
    fields[0] = "12345"
    fields[1] = "ABC"
    fields[21] = "ABC Limited"
    fields[27] = str(int((pd.Timestamp("2024-01-15") - pd.Timestamp("1980-01-01")).total_seconds()))
    fields[53] = "INE123A01016"
    payload = ("NEATCM|01.00.00|0\n" + "|".join(fields) + "\n").encode()
    found = parse_security_master(gzip.compress(payload), {"ABC"},
        "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_15012024.csv.gz")
    assert len(found) == 1
    assert found[0]["listing_date"] == "2024-01-15"
    assert found[0]["company_name"] == "ABC Limited"
    assert found[0]["isin"] == "INE123A01016"


def test_nse_mii_v16_pipe_security_master_uses_specified_fields():
    import gzip
    from src.listing_master import parse_security_master

    # v1.6 record fields after expanding the six market-eligibility values:
    # symbol=1, company name=15, listing date=21, ISIN=47.
    fields = [""] * 48
    fields[0] = "12345"
    fields[1] = "ABC"
    fields[15] = "ABC Limited"
    fields[21] = str(int((pd.Timestamp("2024-01-15") - pd.Timestamp("1980-01-01")).total_seconds()))
    fields[47] = "INE123A01016"
    payload = ("NEATCM|01.00.00|0\n" + "|".join(fields) + "\n").encode()

    found = parse_security_master(gzip.compress(payload), {"ABC"},
        "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_15012024.csv.gz")
    assert len(found) == 1
    assert found[0]["listing_date"] == "2024-01-15"
    assert found[0]["company_name"] == "ABC Limited"
    assert found[0]["isin"] == "INE123A01016"


def test_official_equity_listing_csv_parser_extracts_verified_dates():
    from src.listing_master import parse_equity_listing_csv
    csv = """SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING,PAID UP VALUE,MARKET LOT,ISIN NUMBER,FACE VALUE
AAA,AAA Limited,EQ,06-OCT-2008,5,1,INE144J01016,5
BBB,BBB Limited,BE,03-MAY-95,10,1,INE253B01015,10
OTHER,Other Limited,EQ,01-JAN-2000,10,1,INE000A01000,10
"""
    rows = parse_equity_listing_csv(csv, {"AAA", "BBB"},
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv")
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["AAA"]["listing_date"] == "2008-10-06"
    assert by_symbol["BBB"]["listing_date"] == "1995-05-03"
    assert len(rows) == 2
    assert all("nsearchives.nseindia.com" in row["listing_date_source"] for row in rows)


def test_official_equity_listing_csv_rejects_unknown_schema():
    from src.listing_master import parse_equity_listing_csv
    import pytest
    with pytest.raises(ValueError, match="Symbol/Date of Listing"):
        parse_equity_listing_csv("Ticker,Name\nAAA,AAA Limited\n", {"AAA"}, "official-source")
