from datetime import datetime, timezone
import json

import pandas as pd

from src import model_progress_audit as audit


def _calendar(root):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "nse_market_calendar.json").write_text(json.dumps({
        "schema_version": 1, "exchange": "NSE", "segment": "CM",
        "timezone": "Asia/Kolkata",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holidays": [], "special_sessions": [],
    }), encoding="utf-8")


def _check(rows, name):
    return next(row for row in rows if row["check"] == name)


def test_calendar_missing_refresh_timestamp_is_not_a_pass(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "nse_market_calendar.json").write_text(json.dumps({
        "exchange": "NSE", "segment": "CM", "holidays": [], "special_sessions": [],
        "updated_at_utc": None,
    }), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    payload, expected, detail = audit._calendar_context()
    assert payload is not None
    assert expected is None
    assert "updated_at_utc is missing" in detail


def test_ohlcv_audit_detects_duplicate_keys_and_invalid_bars(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    records = [
        {"date": f"2019-01-{(i % 28) + 1:02d}", "symbol": f"S{i:03d}",
         "open": 10, "high": 12, "low": 8, "close": 11, "volume": 100}
        for i in range(100)
    ]
    records[0] = {"date": "2019-01-01", "symbol": "S000", "open": 10, "high": 9,
                  "low": 8, "close": 11, "volume": 100}
    records[1] = {"date": "2019-01-01", "symbol": "S000", "open": 10, "high": 12,
                  "low": 8, "close": 11, "volume": 100}
    pd.DataFrame(records).to_csv(data / "ohlcv.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "OHLCV completeness and price integrity")
    assert result["status"] == "FAIL"
    assert "duplicate_date_symbol=1" in result["detail"]
    assert "invalid_ohlcv=1" in result["detail"]


def test_index_constituent_audit_detects_bad_weights_and_duplicate_memberships(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame([
        {"index": "NIFTY50", "symbol": "ABC", "as_of_date": "2026-11-16", "weight_pct": 110, "weight_method": "OFFICIAL_NSE_WEIGHT"},
        {"index": "NIFTY50", "symbol": "ABC", "as_of_date": "2026-11-16", "weight_pct": 110, "weight_method": "OFFICIAL_NSE_WEIGHT"},
    ]).to_csv(data / "index_constituents_history.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Index membership and weight integrity")
    assert result["status"] == "FAIL"
    assert "duplicate_memberships=1" in result["detail"]
    assert "out_of_range_weights=2" in result["detail"]



def test_empty_feature_artifact_with_schema_is_review_not_malformed(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame(columns=[
        "date", "symbol", "index", "as_of_date", "weight_pct", "weight_method",
        "index_return_1d_pct", "index_return_5d_pct", "stock_return_1d_pct",
        "rolling_corr_60d", "rolling_beta_60d", "weighted_stock_return_contribution_pct",
    ]).to_csv(data / "index_constituent_features.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Stock-index feature output")
    assert result["status"] == "PASS"
    population = _check(rows, "Stock-index feature population and uniqueness")
    assert population["status"] == "REVIEW"
    assert "zero rows" in population["detail"]


def test_index_refresh_receipt_surfaces_per_index_source_failure(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    (data / "index_constituent_intelligence_audit.json").write_text(json.dumps({
        "updated_at_ist": "2026-11-17T09:00:00+05:30",
        "feature_rows": 0,
        "index_ohlcv": {"NIFTY50": {"status": "PASS"}},
        "constituents": {"NIFTY50": {"status": "SOURCE_UNAVAILABLE"}},
    }), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    source = _check(rows, "Per-index constituent source results")
    assert source["status"] == "REVIEW"
    assert "NIFTY50" in source["detail"]


def test_blocked_gate_alone_keeps_overall_audit_red(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "MODULES", {})
    monkeypatch.setattr(audit, "ARTIFACTS", {})
    monkeypatch.setattr(audit, "PHASE_EVIDENCE", [])
    for rel in ["README.md", ".github/workflows/daily.yml",
                ".github/workflows/weekend-data-audit.yml",
                ".github/workflows/pipeline-integrity-ci.yml"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("present", encoding="utf-8")
    def add_blocked(rows):
        audit.add(rows, "test", "blocked only", "BLOCKED", "required evidence absent")
    monkeypatch.setattr(audit, "audit_live_data", add_blocked)
    result = audit.run_audit(tmp_path / "audit-output")
    report = json.loads((tmp_path / "audit-output/model_progress_audit.json").read_text())
    assert report["counts"]["FAIL"] == 0
    assert report["counts"]["BLOCKED"] > 0
    assert result == 1


def test_listing_master_uses_verified_source_column_name(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame([{
        "symbol": "ABC", "listing_date": "2020-01-01",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=ABC",
        "verified_at_utc": "2026-11-17T00:00:00+00:00",
    }]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "NSE listing-date master")
    assert result["status"] == "PASS"


def test_index_ohlcv_missing_configured_index_is_not_a_pass(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    indexes = sorted({
        "NIFTY50", "NIFTYNEXT50", "NIFTY100", "NIFTY200", "NIFTY500",
        "BANKNIFTY", "NIFTYIT", "NIFTYAUTO", "NIFTYFIN", "NIFTYPHARMA", "NIFTYFMCG",
    })
    pd.DataFrame([{
        "date": "2026-11-16", "index": name, "open": 100, "high": 102,
        "low": 99, "close": 101,
    } for name in indexes[:-1]]).to_csv(data / "index_ohlcv.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Index OHLCV row integrity")
    assert result["status"] == "REVIEW"
    assert indexes[-1] in result["detail"]


def test_constituent_weight_total_shortfall_is_visible(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame([
        {"index": "NIFTY50", "symbol": "AAA", "as_of_date": "2026-11-16",
         "weight_pct": 60, "weight_method": "OFFICIAL_NSE_WEIGHT"},
        {"index": "NIFTY50", "symbol": "BBB", "as_of_date": "2026-11-16",
         "weight_pct": 30, "weight_method": "OFFICIAL_NSE_WEIGHT"},
    ]).to_csv(data / "index_constituents_history.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Constituent weight totals and coverage")
    assert result["status"] == "REVIEW"
    assert "90.0" in result["detail"]


def test_index_features_reject_future_membership_lookahead(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame([{
        "date": "2026-11-16", "as_of_date": "2026-11-17", "symbol": "AAA",
        "index": "NIFTY50", "weight_pct": 2.0, "weight_method": "OFFICIAL_NSE_WEIGHT",
        "index_return_1d_pct": 0.5, "index_return_5d_pct": 1.2,
        "stock_return_1d_pct": 0.8, "rolling_corr_60d": 0.7,
        "rolling_beta_60d": 1.1, "weighted_stock_return_contribution_pct": 0.016,
    }]).to_csv(data / "index_constituent_features.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Stock-index feature population, fields and no-lookahead integrity")
    assert result["status"] == "FAIL"
    assert "future_membership_rows=1" in result["detail"]


def test_listing_master_rejects_blank_invalid_and_non_nse_metadata(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame([{
        "symbol": "AAA", "listing_date": "not-a-date",
        "listing_date_source": "https://example.com/fake",
        "verified_at_utc": "2026-11-16T00:00:00Z",
    }]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Listing master field completeness and validity")
    assert result["status"] == "FAIL"
    assert "invalid_listing_dates=1" in result["detail"]
    assert "non_NSE_source_urls=1" in result["detail"]


def test_calendar_rejects_duplicate_special_sessions(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "nse_market_calendar.json").write_text(json.dumps({
        "exchange": "NSE", "segment": "CM",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holidays": [],
        "special_sessions": [
            {"date": "2026-11-08", "status": "CONFIRMED"},
            {"date": "2026-11-08", "status": "CONFIRMED"},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Calendar holiday and special-session records")
    assert result["status"] == "FAIL"
    assert "duplicate_special_dates=True" in result["detail"]


def test_ohlcv_bars_before_verified_listing_date_fail_audit(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    dates = pd.date_range("2019-01-01", periods=100, freq="D")
    pd.DataFrame([{
        "date": day, "symbol": "AAA", "open": 10, "high": 12,
        "low": 9, "close": 11, "volume": 100,
    } for day in dates]).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{
        "symbol": "AAA", "listing_date": "2020-01-01",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=AAA",
        "verified_at_utc": "2026-11-16T00:00:00Z",
    }]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "OHLCV does not precede verified listing date")
    assert result["status"] == "FAIL"
    assert "pre_listing_bars=100" in result["detail"]


def test_empty_listing_master_schema_is_review_not_missing_artifact(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    pd.DataFrame(columns=["symbol", "company_name", "isin", "listing_date",
        "listing_date_source", "verified_at_utc"]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Listing master field completeness and validity")
    assert result["status"] == "REVIEW"
    assert "no verified listing-date rows" in result["detail"]


def test_listing_master_accepts_official_nse_hosts_and_rejects_lookalikes(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    sources = [
        "https://www.nseindia.com/api/quote-equity?symbol=AAA",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://example.com/nseindia.com/fake",
        "https://nseindia.com.attacker.example/fake",
        "http://www.nseindia.com/insecure",
        "not a url",
    ]
    pd.DataFrame([{
        "symbol": f"S{i}", "listing_date": "2020-01-01",
        "listing_date_source": source,
        "verified_at_utc": "2026-11-16T00:00:00Z",
    } for i, source in enumerate(sources)]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "Listing master field completeness and validity")
    assert result["status"] == "FAIL"
    assert "non_NSE_source_urls=4" in result["detail"]


def test_pre_listing_audit_reports_per_symbol_counts_and_date_ranges(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    # The live-data audit intentionally requires at least 100 OHLCV rows.
    # Keep this fixture above that production threshold while checking the
    # per-symbol pre-listing counts and date ranges.
    dates = pd.date_range("2019-01-01", periods=50, freq="D")
    pd.DataFrame([{
        "date": day, "symbol": symbol, "open": 10, "high": 12,
        "low": 9, "close": 11, "volume": 100,
    } for symbol in ["AAA", "BBB"] for day in dates]).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{
        "symbol": symbol, "listing_date": "2020-01-01",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=" + symbol,
        "verified_at_utc": "2026-11-16T00:00:00Z",
    } for symbol in ["AAA", "BBB"]]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "OHLCV does not precede verified listing date")
    assert result["status"] == "FAIL"
    assert "pre_listing_bars=100" in result["detail"]
    assert "AAA:count=50,first=2019-01-01,last=2019-02-19,listing=2020-01-01" in result["detail"]
    assert "BBB:count=50,first=2019-01-01,last=2019-02-19,listing=2020-01-01" in result["detail"]


def test_prelisting_bars_with_symbol_provenance_case_are_review_not_false_failure(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    dates = pd.date_range("2019-01-01", periods=100, freq="D")
    pd.DataFrame([{
        "date": day, "symbol": "AAA", "open": 10, "high": 12,
        "low": 9, "close": 11, "volume": 100,
    } for day in dates]).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{
        "symbol": "AAA", "listing_date": "2020-01-01",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=AAA",
        "verified_at_utc": "2026-11-16T00:00:00Z",
    }]).to_csv(data / "nse_listing_master.csv", index=False)
    pd.DataFrame([{
        "symbol": "AAA", "event_type": "BSE_HISTORY_REVIEW",
        "predecessor_symbol": "", "evidence_start_date": "", "evidence_end_date": "",
        "evidence_exchange": "BSE", "evidence_url": "https://example.com/evidence",
        "evidence_note": "Case registered; individual bar source still requires verification.",
        "classification_status": "ROW_SOURCE_UNVERIFIED",
    }]).to_csv(data / "historical_listing_provenance.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "OHLCV does not precede verified listing date")
    assert result["status"] == "REVIEW"
    assert "provenance_case_coverage=1/1" in result["detail"]
    assert "row_level_exchange_attribution_unresolved=True" in result["detail"]


def test_prelisting_bars_without_provenance_case_remain_failure(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    dates = pd.date_range("2019-01-01", periods=100, freq="D")
    pd.DataFrame([{
        "date": day, "symbol": "AAA", "open": 10, "high": 12,
        "low": 9, "close": 11, "volume": 100,
    } for day in dates]).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{
        "symbol": "AAA", "listing_date": "2020-01-01",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=AAA",
        "verified_at_utc": "2026-11-16T00:00:00Z",
    }]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    rows = []
    audit.audit_live_data(rows)
    result = _check(rows, "OHLCV does not precede verified listing date")
    assert result["status"] == "FAIL"

def test_listing_date_parser_handles_day_first_format_without_warnings(tmp_path, monkeypatch):
    import warnings

    _calendar(tmp_path)
    data = tmp_path / "data"
    dates = pd.date_range("2019-01-01", periods=100, freq="D")
    pd.DataFrame([{
        "date": day, "symbol": "AAA", "open": 10, "high": 12,
        "low": 9, "close": 11, "volume": 100,
    } for day in dates]).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{
        "symbol": "AAA", "listing_date": "31/12/2019",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=AAA",
        "verified_at_utc": "2026-11-16T00:00:00Z",
    }]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    rows = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit.audit_live_data(rows)

    listing = _check(rows, "Listing master field completeness and validity")
    prelisting = _check(rows, "OHLCV does not precede verified listing date")
    assert listing["status"] == "PASS"
    assert prelisting["status"] == "FAIL"
    assert "pre_listing_bars=100" in prelisting["detail"]
    assert not [warning for warning in caught if "dayfirst" in str(warning.message).lower()]

def test_prelisting_audit_reports_every_affected_symbol_not_only_top_twenty(tmp_path, monkeypatch):
    _calendar(tmp_path)
    data = tmp_path / "data"
    symbols = [f"S{i:02d}" for i in range(25)]
    dates = pd.date_range("2019-01-01", periods=4, freq="D")
    bars = [{
        "date": day, "symbol": symbol, "open": 10, "high": 12,
        "low": 9, "close": 11, "volume": 100,
    } for symbol in symbols for day in dates]
    pd.DataFrame(bars).to_csv(data / "ohlcv.csv", index=False)
    pd.DataFrame([{
        "symbol": symbol, "listing_date": "2020-01-01",
        "listing_date_source": "https://www.nseindia.com/api/quote-equity?symbol=" + symbol,
        "verified_at_utc": "2026-11-16T00:00:00Z",
    } for symbol in symbols]).to_csv(data / "nse_listing_master.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    rows = []
    audit.audit_live_data(rows)

    result = _check(rows, "OHLCV does not precede verified listing date")
    assert result["status"] == "FAIL"
    assert "affected_symbols=25" in result["detail"]
    assert "by_symbol_all=" in result["detail"]
    assert all(f"{symbol}:count=4" in result["detail"] for symbol in symbols)
