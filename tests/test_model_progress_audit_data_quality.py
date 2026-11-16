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
