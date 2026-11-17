import pandas as pd

from src.historical_listing_provenance import audit_historical_listing_provenance


def _inputs(root, include_source=False):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    if include_source:
        columns.append("source")
    rows = []
    for day in pd.date_range("2020-09-28", periods=10, freq="D"):
        item = {"date": day, "symbol": "CAMS", "open": 10, "high": 11,
                "low": 9, "close": 10, "volume": 100}
        if include_source:
            item["source"] = "Yahoo Finance"
        rows.append(item)
    pd.DataFrame(rows, columns=columns).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{"symbol": "CAMS", "listing_date": "2021-05-07"}]).to_csv(
        data / "nse_listing_master.csv", index=False
    )
    pd.DataFrame([{
        "symbol": "CAMS", "event_type": "BSE_LISTING_AND_NSE_PERMITTED_TO_TRADE",
        "predecessor_symbol": "", "evidence_start_date": "2020-10-01",
        "evidence_end_date": "2021-05-06", "evidence_exchange": "BSE;NSE",
        "evidence_url": "https://www.camsonline.com/official.pdf",
        "evidence_note": "BSE listing before NSE listing",
        "classification_status": "ROW_SOURCE_UNVERIFIED",
    }]).to_csv(data / "historical_listing_provenance.csv", index=False)


def test_prelisting_provenance_audit_preserves_and_reports_rows(tmp_path):
    _inputs(tmp_path)
    result = audit_historical_listing_provenance(tmp_path)
    assert result["status"] == "REVIEW"
    assert result["pre_listing_bars"] == 10
    assert result["bars_inside_documented_event_windows"] == 7
    assert "OHLCV_exchange_or_provider_fields=absent" in result["detail"]
    assert "no OHLCV rows changed or deleted" in result["detail"]
    assert len(pd.read_csv(tmp_path / "data" / "ohlcv.csv")) == 10


def test_event_window_does_not_claim_row_level_exchange_attribution(tmp_path):
    _inputs(tmp_path)
    history_path = tmp_path / "data" / "ohlcv.csv"
    history = pd.read_csv(history_path)
    history["date"] = pd.date_range("2020-10-01", periods=10, freq="D")
    history.to_csv(history_path, index=False)
    result = audit_historical_listing_provenance(tmp_path)
    assert result["pre_listing_bars"] == 10
    assert result["bars_inside_documented_event_windows"] == 10
    assert result["status"] == "REVIEW"
    assert "row_level_attribution" in result["detail"]


def test_provenance_audit_blocks_when_listing_master_is_unavailable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(columns=[
        "symbol", "event_type", "predecessor_symbol", "evidence_start_date",
        "evidence_end_date", "evidence_exchange", "evidence_url", "evidence_note",
        "classification_status",
    ]).to_csv(data / "historical_listing_provenance.csv", index=False)
    result = audit_historical_listing_provenance(tmp_path)
    assert result["status"] == "BLOCKED"
    assert "nse_listing_master.csv" in result["detail"]

def test_provenance_listing_date_parser_accepts_day_first_dates(tmp_path):
    _inputs(tmp_path)
    listing_path = tmp_path / "data" / "nse_listing_master.csv"
    pd.DataFrame([{"symbol": "CAMS", "listing_date": "31/05/2021"}]).to_csv(
        listing_path, index=False
    )

    result = audit_historical_listing_provenance(tmp_path)

    assert result["status"] == "REVIEW"
    assert result["pre_listing_bars"] == 10
    assert result["bars_inside_documented_event_windows"] == 7




def test_provenance_audit_fails_when_affected_symbol_is_missing_from_registry(tmp_path):
    _inputs(tmp_path)
    data = tmp_path / "data"
    history_path = data / "ohlcv.csv"
    history = pd.read_csv(history_path)
    extra = history.copy()
    extra["symbol"] = "MISSING"
    history = pd.concat([history, extra], ignore_index=True)
    history.to_csv(history_path, index=False)

    listings_path = data / "nse_listing_master.csv"
    listings = pd.read_csv(listings_path)
    listings = pd.concat([
        listings,
        pd.DataFrame([{"symbol": "MISSING", "listing_date": "2021-05-07"}]),
    ], ignore_index=True)
    listings.to_csv(listings_path, index=False)

    result = audit_historical_listing_provenance(tmp_path)

    assert result["status"] == "FAIL"
    assert result["affected_symbols"] == ["CAMS", "MISSING"]
    assert result["missing_registry_symbols"] == ["MISSING"]
    assert "missing_registry_symbols=['MISSING']" in result["detail"]
    # The audit is read-only: both symbols' original OHLCV rows remain present.
    assert len(pd.read_csv(history_path)) == 20
