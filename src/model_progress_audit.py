"""Evidence-based, failure-isolated progress audit for the Stock Picker repository.

This is an inventory/diagnostic tool. File presence is never treated as proof of
model quality or production readiness; evidence statuses are surfaced verbatim.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "Universe management": "src/universe.py",
    "Core forecasting pipeline": "src/pipeline.py",
    "Market calendar": "src/market_calendar.py",
    "Listing-date master": "src/listing_master.py",
    "Historical coverage audit": "src/historical_coverage_audit.py",
    "Historical gap recovery": "src/repair_historical_gaps.py",
    "Index OHLCV and constituents": "src/index_constituent_intelligence.py",
    "Market/index regime intelligence": "src/index_intelligence.py",
    "News context": "src/news_context.py",
    "Ranking model": "src/ranking_model.py",
    "Directional model": "src/directional_model.py",
    "Confidence model": "src/confidence.py",
    "Trade-quality model": "src/trade_quality_model.py",
    "Risk management": "src/risk_management.py",
    "Paper trading V1": "src/paper_trading.py",
    "Paper trading V2": "src/paper_trading_v2.py",
    "Realized outcomes": "src/outcomes.py",
    "Model governance": "src/model_governance.py",
    "System health": "src/system_health.py",
    "Performance audit": "src/performance_audit.py",
}

ARTIFACTS = {
    "NSE universe": ("data/universe.csv", {"symbol"}),
    "Historical OHLCV": ("data/ohlcv.csv", {"date", "symbol", "open", "high", "low", "close", "volume"}),
    "Predictions": ("data/predictions.csv", {"symbol"}),
    "Prediction evaluations": ("data/evaluations.csv", set()),
    "Candidate history": ("data/prediction_candidates_history.csv", set()),
    "Paper trades V1": ("data/paper_trades.csv", set()),
    "Paper trades V2": ("data/paper_trades_v2.csv", set()),
    "NSE market calendar": ("data/nse_market_calendar.json", None),
    "Historical coverage summary": ("data/historical_coverage_audit.json", None),
    "Phase 107 P&L reconciliation": ("data/phase107_pnl_reconciliation_summary.json", None),
    "Phase 114 promotion readiness": ("data/phase114_promotion_readiness_summary.json", None),
    "Model governance": ("data/model_governance.json", None),
    "Index OHLCV": ("data/index_ohlcv.csv", set()),
    "Index constituent history": ("data/index_constituents_history.csv", set()),
    "Index constituent features": ("data/index_constituent_features.csv", set()),
    "Index intelligence audit": ("data/index_constituent_intelligence_audit.json", None),
    "Weekend data-quality audit": ("data/weekend_data_quality_audit.json", None),
    "Listing master": ("data/nse_listing_master.csv", set()),
}

PHASE_EVIDENCE = [
    "data/phase20_accuracy_summary.json", "data/phase21_26_summary.json",
    "data/phase27_data_quality_summary.json", "data/phase28_30_summary.json",
    "data/phase31_summary.json", "data/phase41_43_summary.json",
    "data/phase48_52_summary.json", "data/phase58_62_summary.json",
    "data/phase63_72_readiness_summary.json", "data/phase73_76_diagnostics.json",
    "data/phase77_80_diagnostics.json", "data/phase81_84_diagnostics.json",
    "data/phase85_88_diagnostics.json", "data/phase89_92_diagnostics.json",
    "data/phase93_96_diagnostics.json", "data/phase97_forecast_quality_summary.json",
    "data/phase98_101_quality_summary.json", "data/phase103_forecast_evidence_summary.json",
    "data/phase107_pnl_reconciliation_summary.json", "data/phase110_forecast_accuracy_summary.json",
    "data/phase111_chronological_replay_summary.json", "data/phase112_market_intelligence_summary.json",
    "data/phase113_sustained_evidence_summary.json", "data/phase114_promotion_readiness_summary.json",
    "data/phase115_shadow_monitoring_summary.json", "data/phase120_forecast_stability_summary.json",
    "data/phase122_challenger_decision.json", "data/phase123_magnitude_calibration_decision.json",
    "data/phase124_feature_signal_decision.json", "data/phase125_feature_ablation_decision.json",
    "data/phase126_objective_target_decision.json",
]

def add(rows, group, name, status, detail, evidence=""):
    rows.append({"group": group, "check": name, "status": status, "detail": str(detail), "evidence": evidence})

def inspect_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, (dict, list)):
        raise ValueError("top-level JSON must be an object or array")
    return value


def _read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)

def _calendar_context():
    """Return the persisted NSE CM calendar and latest completed expected session."""
    from datetime import date, datetime, time, timedelta
    from zoneinfo import ZoneInfo
    calendar_path = ROOT / "data/nse_market_calendar.json"
    if not calendar_path.exists():
        return None, None, "calendar file missing"
    payload = inspect_json(calendar_path)
    if not isinstance(payload, dict) or payload.get("exchange") != "NSE" or payload.get("segment") != "CM":
        return payload, None, "calendar exchange/segment metadata invalid"
    updated = payload.get("updated_at_utc")
    if not updated:
        return payload, None, "calendar updated_at_utc is missing"
    try:
        refreshed = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
        if refreshed.tzinfo is None:
            refreshed = refreshed.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - refreshed.astimezone(timezone.utc)).total_seconds() / 86400
        if age_days > 10:
            return payload, None, f"calendar refresh is {age_days:.1f} days old (>10)"
    except Exception as exc:
        return payload, None, f"calendar updated_at_utc is invalid: {exc}"
    try:
        from src.market_calendar import session_for, IST
        now = datetime.now(IST)
        candidate = now.date()
        if now.time() < time(15, 45):
            candidate -= timedelta(days=1)
        for _ in range(370):
            state = session_for(candidate, payload)
            if state.get("is_open"):
                return payload, candidate.isoformat(), f"calendar refreshed; latest completed expected session {candidate.isoformat()}"
            candidate -= timedelta(days=1)
    except Exception as exc:
        return payload, None, f"could not calculate expected session: {exc}"
    return payload, None, "no expected session found in calendar horizon"

def _check_csv_nonempty(rows, label, rel, required, min_rows=1):
    path = ROOT / rel
    if not path.exists():
        add(rows, "Live data quality and freshness", label, "BLOCKED", "File missing; data-quality validation cannot run", rel)
        return None
    try:
        frame = _read_csv(path)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            add(rows, "Live data quality and freshness", label, "FAIL", f"Missing required columns: {missing}", rel)
            return None
        if len(frame) < min_rows:
            add(rows, "Live data quality and freshness", label, "FAIL", f"Only {len(frame)} rows; minimum is {min_rows}", rel)
            return None
        add(rows, "Live data quality and freshness", label, "PASS", f"{len(frame):,} rows; required columns present", rel)
        return frame
    except Exception as exc:
        add(rows, "Live data quality and freshness", label, "FAIL", f"CSV could not be read: {type(exc).__name__}: {exc}", rel)
        return None

def audit_live_data(rows):
    """Validate populated fields, key uniqueness, temporal freshness and update coverage."""
    import numpy as np
    import pandas as pd

    calendar, expected_session, calendar_detail = _calendar_context()
    cal_status = "PASS" if expected_session else ("FAIL" if "missing" in calendar_detail or "invalid" in calendar_detail or "old" in calendar_detail else "REVIEW")
    add(rows, "Live data quality and freshness", "Official NSE calendar refresh",
        cal_status, calendar_detail, "data/nse_market_calendar.json")
    if calendar is not None:
        holidays = calendar.get("holidays", [])
        specials = calendar.get("special_sessions", [])
        valid_holidays = isinstance(holidays, list) and all(isinstance(x, dict) and x.get("date") for x in holidays)
        valid_specials = isinstance(specials, list) and all(
            isinstance(x, dict) and x.get("date") and x.get("status", "CONFIRMED") in {"CONFIRMED", "PROPOSED"}
            for x in specials
        )
        dates = [x.get("date") for x in holidays if isinstance(x, dict)]
        duplicate_dates = len(dates) != len(set(dates))
        result = "PASS" if valid_holidays and valid_specials and not duplicate_dates else "FAIL"
        add(rows, "Live data quality and freshness", "Calendar holiday and special-session records",
            result, f"holidays={len(holidays) if isinstance(holidays, list) else 'invalid'}; special_sessions={len(specials) if isinstance(specials, list) else 'invalid'}; duplicate_holiday_dates={duplicate_dates}",
            "data/nse_market_calendar.json")
        audit_path = ROOT / "data/nse_market_calendar_audit.json"
        if audit_path.exists():
            try:
                ca = inspect_json(audit_path)
                st = str(ca.get("status", "")).upper()
                add(rows, "Live data quality and freshness", "Calendar refresh receipt",
                    "PASS" if st in {"REFRESHED", "PASS", "OK"} else "FAIL",
                    f"refresh receipt status={st or 'missing'}; refreshed_at={ca.get('refreshed_at_utc')}",
                    "data/nse_market_calendar_audit.json")
            except Exception as exc:
                add(rows, "Live data quality and freshness", "Calendar refresh receipt", "FAIL", str(exc), "data/nse_market_calendar_audit.json")
        else:
            add(rows, "Live data quality and freshness", "Calendar refresh receipt", "BLOCKED", "No source-refresh receipt exists", "data/nse_market_calendar_audit.json")

    universe = _check_csv_nonempty(rows, "Universe population and keys", "data/universe.csv", {"symbol"}, 100)
    if universe is not None:
        symbols = universe["symbol"].fillna("").astype(str).str.strip().str.upper()
        duplicates = int(symbols.duplicated().sum())
        blanks = int(symbols.eq("").sum())
        result = "PASS" if duplicates == 0 and blanks == 0 else "FAIL"
        add(rows, "Live data quality and freshness", "Universe symbols populated and unique",
            result, f"symbols={len(symbols)}; blank={blanks}; duplicate={duplicates}", "data/universe.csv")
        universe_symbols = set(symbols[symbols.ne("") & ~symbols.str.startswith("DUMMY")])
    else:
        universe_symbols = set()

    history = _check_csv_nonempty(rows, "OHLCV populated values and schema", "data/ohlcv.csv",
        {"date", "symbol", "open", "high", "low", "close", "volume"}, 100)
    if history is not None:
        h = history.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            h[col] = pd.to_numeric(h[col], errors="coerce")
        h["symbol"] = h["symbol"].fillna("").astype(str).str.strip().str.upper()
        missing_cells = int(h[["date", "symbol", "open", "high", "low", "close", "volume"]].isna().sum().sum())
        duplicate_keys = int(h.duplicated(["date", "symbol"]).sum())
        valid = h[["open", "high", "low", "close"]].gt(0).all(axis=1) & h["volume"].ge(0)
        valid &= h["high"].ge(h[["open", "close"]].max(axis=1)) & h["low"].le(h[["open", "close"]].min(axis=1))
        invalid_rows = int((~valid & h[["open", "high", "low", "close", "volume"]].notna().all(axis=1)).sum())
        future_rows = int((h["date"].dt.date > datetime.now(timezone.utc).date()).sum())
        result = "PASS" if missing_cells == 0 and duplicate_keys == 0 and invalid_rows == 0 and future_rows == 0 else "FAIL"
        add(rows, "Live data quality and freshness", "OHLCV completeness and price integrity", result,
            f"rows={len(h):,}; missing_required_cells={missing_cells}; duplicate_date_symbol={duplicate_keys}; invalid_ohlcv={invalid_rows}; future_dated_rows={future_rows}",
            "data/ohlcv.csv")
        latest = h["date"].max()
        latest_day = latest.date().isoformat() if pd.notna(latest) else "unavailable"
        age_status = "REVIEW" if not expected_session else ("PASS" if latest_day >= expected_session else "FAIL")
        add(rows, "Live data quality and freshness", "OHLCV latest expected session coverage", age_status,
            f"latest_ohlcv_date={latest_day}; latest_expected_completed_session={expected_session or 'unknown'}; {calendar_detail}",
            "data/ohlcv.csv")
        if universe_symbols:
            observed_symbols = set(h.loc[h["date"].eq(latest), "symbol"].dropna())
            missing_symbols = sorted(universe_symbols - observed_symbols)
            add(rows, "Live data quality and freshness", "Latest OHLCV universe coverage",
                "PASS" if not missing_symbols else "REVIEW",
                f"latest_date_symbols={len(observed_symbols)}/{len(universe_symbols)}; missing={len(missing_symbols)}; examples={missing_symbols[:15]}",
                "data/ohlcv.csv")
    else:
        add(rows, "Live data quality and freshness", "OHLCV latest expected session coverage", "BLOCKED", "OHLCV validation unavailable", "data/ohlcv.csv")

    fundamentals = _check_csv_nonempty(rows, "Fundamentals population and freshness", "data/fundamentals.csv",
        {"symbol", "updated_at"}, 50)
    if fundamentals is not None:
        f = fundamentals.copy()
        f["updated_at"] = pd.to_datetime(f["updated_at"], errors="coerce", utc=True)
        latest = f["updated_at"].max()
        latest_text = latest.isoformat() if pd.notna(latest) else "unavailable"
        stale = int(f["updated_at"].isna().sum())
        age = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 86400 if pd.notna(latest) else 9999
        add(rows, "Live data quality and freshness", "Fundamentals update timestamps",
            "PASS" if stale == 0 and age <= 10 else "REVIEW",
            f"rows={len(f)}; missing_timestamps={stale}; newest={latest_text}; newest_age_days={age:.1f}",
            "data/fundamentals.csv")
        if universe_symbols:
            fsyms = set(f["symbol"].dropna().astype(str).str.upper().str.strip())
            missing = sorted(universe_symbols - fsyms)
            add(rows, "Live data quality and freshness", "Fundamentals universe coverage",
                "PASS" if not missing else "REVIEW",
                f"covered={len(universe_symbols)-len(missing)}/{len(universe_symbols)}; missing={len(missing)}; examples={missing[:15]}",
                "data/fundamentals.csv")
        numeric_fields = [c for c in f.columns if c not in {"symbol", "updated_at", "sector"}]
        if numeric_fields:
            missing_pct = f[numeric_fields].apply(pd.to_numeric, errors="coerce").isna().mean().sort_values(ascending=False)
            top = ", ".join(f"{k}={v:.0%}" for k, v in missing_pct.head(8).items())
            add(rows, "Live data quality and freshness", "Fundamental field population",
                "REVIEW" if (missing_pct > 0.5).any() else "PASS",
                f"missing percentages by field (top): {top}; nulls can be legitimate when source does not publish a metric",
                "data/fundamentals.csv")

    listing = _check_csv_nonempty(rows, "NSE listing-date master", "data/nse_listing_master.csv",
        {"symbol", "listing_date", "source", "verified_at_utc"}, 1)
    if listing is not None:
        required = ["symbol", "listing_date", "source", "verified_at_utc"]
        nulls = int(listing[required].isna().sum().sum())
        unique = int(listing["symbol"].astype(str).str.upper().str.strip().duplicated().sum())
        add(rows, "Live data quality and freshness", "Listing master field completeness",
            "PASS" if nulls == 0 and unique == 0 else "FAIL",
            f"rows={len(listing)}; missing_required_cells={nulls}; duplicate_symbols={unique}",
            "data/nse_listing_master.csv")
        if universe_symbols:
            lsyms = set(listing["symbol"].dropna().astype(str).str.upper().str.strip())
            missing = sorted(universe_symbols - lsyms)
            add(rows, "Live data quality and freshness", "Listing-date master universe coverage",
                "PASS" if not missing else "REVIEW",
                f"covered={len(universe_symbols)-len(missing)}/{len(universe_symbols)}; unresolved={len(missing)}; examples={missing[:15]}",
                "data/nse_listing_master.csv")

    index_bars = _check_csv_nonempty(rows, "Index OHLCV populated values", "data/index_ohlcv.csv",
        {"date", "index", "open", "high", "low", "close"}, 1)
    expected_indexes = {"NIFTY50", "NIFTYNEXT50", "NIFTY100", "NIFTY200", "NIFTY500",
                        "BANKNIFTY", "NIFTYIT", "NIFTYAUTO", "NIFTYFIN", "NIFTYPHARMA", "NIFTYFMCG"}
    if index_bars is not None:
        ib = index_bars.copy()
        ib["date"] = pd.to_datetime(ib["date"], errors="coerce")
        for c in ["open", "high", "low", "close"]:
            ib[c] = pd.to_numeric(ib[c], errors="coerce")
        ib["index"] = ib["index"].fillna("").astype(str).str.upper().str.strip()
        invalid = int((ib[["date", "open", "high", "low", "close"]].isna().any(axis=1) |
                       ~ib[["open", "high", "low", "close"]].gt(0).all(axis=1) |
                       ~ib["high"].ge(ib[["open", "close"]].max(axis=1)) |
                       ~ib["low"].le(ib[["open", "close"]].min(axis=1))).sum())
        dup = int(ib.duplicated(["date", "index"]).sum())
        missing_idx = sorted(expected_indexes - set(ib["index"]))
        result = "PASS" if invalid == 0 and dup == 0 else "FAIL"
        add(rows, "Live data quality and freshness", "Index OHLCV row integrity", result,
            f"rows={len(ib)}; invalid_rows={invalid}; duplicate_index_dates={dup}; indexes_without_history={missing_idx}",
            "data/index_ohlcv.csv")
        latest = ib.groupby("index")["date"].max()
        stale = {k: str(v.date()) for k, v in latest.items()
                 if expected_session and v.date().isoformat() < expected_session}
        absent = sorted(expected_indexes - set(latest.index))
        add(rows, "Live data quality and freshness", "Index OHLCV update coverage",
            "PASS" if not absent and not stale else "REVIEW",
            f"expected_indexes={len(expected_indexes)}; present={len(latest)}; absent={absent}; stale_by_index={stale}; expected_session={expected_session}",
            "data/index_ohlcv.csv")
    else:
        add(rows, "Live data quality and freshness", "Index OHLCV update coverage", "BLOCKED", "No index bars to verify; data collection workflow may not have run", "data/index_ohlcv.csv")

    # The refresh receipt is authoritative for source health. A successful Python
    # process can still contain SOURCE_UNAVAILABLE results for individual indices.
    index_audit_path = ROOT / "data/index_constituent_intelligence_audit.json"
    index_audit = None
    if index_audit_path.exists():
        try:
            index_audit = inspect_json(index_audit_path)
            index_source = index_audit.get("index_ohlcv", {})
            member_source = index_audit.get("constituents", {})
            expected_source_names = set(expected_indexes)
            failed_bars = {k: v for k, v in index_source.items() if v.get("status") != "PASS"}
            failed_members = {k: v for k, v in member_source.items() if v.get("status") != "PASS"}
            add(rows, "Live source refresh receipts", "Per-index OHLCV source results",
                "PASS" if not failed_bars and expected_source_names.issubset(index_source) else "REVIEW",
                f"configured={len(expected_source_names)}; reported={len(index_source)}; failed_or_missing={list(failed_bars) + sorted(expected_source_names-set(index_source))}; results={ {k:v.get('status') for k,v in index_source.items()} }",
                "data/index_constituent_intelligence_audit.json")
            add(rows, "Live source refresh receipts", "Per-index constituent source results",
                "PASS" if not failed_members and expected_source_names.issubset(member_source) else "REVIEW",
                f"configured={len(expected_source_names)}; reported={len(member_source)}; failed_or_missing={list(failed_members) + sorted(expected_source_names-set(member_source))}; results={ {k:v.get('status') for k,v in member_source.items()} }",
                "data/index_constituent_intelligence_audit.json")
            add(rows, "Live source refresh receipts", "Index refresh report timestamp and feature count",
                "PASS" if index_audit.get("updated_at_ist") and int(index_audit.get("feature_rows", 0)) > 0 else "REVIEW",
                f"updated_at_ist={index_audit.get('updated_at_ist')}; feature_rows={index_audit.get('feature_rows', 0)}",
                "data/index_constituent_intelligence_audit.json")
        except Exception as exc:
            add(rows, "Live source refresh receipts", "Index source refresh receipt parsing", "FAIL", str(exc),
                "data/index_constituent_intelligence_audit.json")
    else:
        add(rows, "Live source refresh receipts", "Index source refresh receipt parsing", "BLOCKED",
            "The index refresher did not produce its per-source audit receipt",
            "data/index_constituent_intelligence_audit.json")

    constituents = _check_csv_nonempty(rows, "Index constituent snapshot population", "data/index_constituents_history.csv",
        {"index", "symbol", "as_of_date", "weight_pct", "weight_method"}, 0)
    if constituents is not None and constituents.empty:
        add(rows, "Live data quality and freshness", "Index constituent snapshot population",
            "REVIEW", "Refresh ran but produced zero constituent rows; inspect per-index NSE source statuses before trusting any membership features",
            "data/index_constituents_history.csv")
    if constituents is not None and not constituents.empty:
        c = constituents.copy()
        c["as_of_date"] = pd.to_datetime(c["as_of_date"], errors="coerce")
        c["weight_pct"] = pd.to_numeric(c["weight_pct"], errors="coerce")
        c["index"] = c["index"].fillna("").astype(str).str.upper().str.strip()
        c["symbol"] = c["symbol"].fillna("").astype(str).str.upper().str.strip()
        dup = int(c.duplicated(["index", "symbol", "as_of_date"]).sum())
        bad_weight = int((c["weight_pct"].notna() & ~c["weight_pct"].between(0, 100)).sum())
        latest_by_index = c.groupby("index")["as_of_date"].max()
        absent = sorted(expected_indexes - set(latest_by_index.index))
        sums = c.dropna(subset=["weight_pct"]).groupby(["index", "as_of_date"])["weight_pct"].sum()
        off_total = {f"{idx}@{day.date()}": round(float(v), 2) for (idx, day), v in sums.items() if abs(v - 100) > 2.0}
        result = "PASS" if dup == 0 and bad_weight == 0 else "FAIL"
        add(rows, "Live data quality and freshness", "Index membership and weight integrity", result,
            f"rows={len(c)}; duplicate_memberships={dup}; out_of_range_weights={bad_weight}; groups_not_approximately_100pct={len(off_total)}; examples={dict(list(off_total.items())[:8])}",
            "data/index_constituents_history.csv")
        stale = {k: str(v.date()) for k, v in latest_by_index.items()
                 if expected_session and v.date().isoformat() < expected_session}
        add(rows, "Live data quality and freshness", "Constituent snapshot update coverage",
            "PASS" if not absent and not stale else "REVIEW",
            f"indexes_present={len(latest_by_index)}/{len(expected_indexes)}; absent={absent}; stale={stale}; expected_session={expected_session}",
            "data/index_constituents_history.csv")
        methods = sorted(c["weight_method"].dropna().astype(str).unique().tolist())
        missing_weight = int(c["weight_pct"].isna().sum())
        add(rows, "Live data quality and freshness", "Official versus estimated constituent weights",
            "REVIEW" if missing_weight else "PASS",
            f"missing_weights={missing_weight}; methods={methods}; estimated values must remain explicitly labelled",
            "data/index_constituents_history.csv")

    features = _check_csv_nonempty(rows, "Stock-index feature output", "data/index_constituent_features.csv",
        {"date", "symbol", "index"}, 0)
    if features is not None and features.empty:
        add(rows, "Live data quality and freshness", "Stock-index feature population and uniqueness",
            "REVIEW", "Feature artifact has a valid schema but zero rows; point-in-time membership or index history is not yet available",
            "data/index_constituent_features.csv")
    if features is not None and not features.empty:
        x = features.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        dup = int(x.duplicated(["date", "symbol", "index"]).sum())
        latest = x["date"].max()
        numeric = [col for col in x.columns if col not in {"date", "symbol", "index"}]
        null_pct = x[numeric].apply(pd.to_numeric, errors="coerce").isna().mean() if numeric else pd.Series(dtype=float)
        add(rows, "Live data quality and freshness", "Stock-index feature population and uniqueness",
            "PASS" if dup == 0 else "FAIL",
            f"rows={len(x)}; duplicate_keys={dup}; latest={latest.date() if pd.notna(latest) else 'unknown'}; feature_null_rates={null_pct.round(3).to_dict()}",
            "data/index_constituent_features.csv")
        if expected_session:
            latest_text = latest.date().isoformat() if pd.notna(latest) else ""
            add(rows, "Live data quality and freshness", "Stock-index feature freshness",
                "PASS" if latest_text >= expected_session else "REVIEW",
                f"latest_feature_date={latest_text}; expected_session={expected_session}",
                "data/index_constituent_features.csv")

    predictions = _check_csv_nonempty(rows, "Prediction output populated fields", "data/predictions.csv",
        {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "rank", "score"}, 1)
    if predictions is not None:
        p = predictions.copy()
        required = ["prediction_date", "target_date", "symbol", "base_close", "predicted_close", "rank", "score"]
        nulls = int(p[required].isna().sum().sum())
        p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce")
        p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce")
        p["base_close"] = pd.to_numeric(p["base_close"], errors="coerce")
        p["predicted_close"] = pd.to_numeric(p["predicted_close"], errors="coerce")
        p["rank"] = pd.to_numeric(p["rank"], errors="coerce")
        p["score"] = pd.to_numeric(p["score"], errors="coerce")
        dup = int(p.duplicated(["prediction_date", "target_date", "symbol"]).sum())
        invalid = int((p["base_close"].le(0) | p["predicted_close"].le(0) | p["rank"].le(0) |
                       p["prediction_date"].isna() | p["target_date"].isna()).sum())
        add(rows, "Live data quality and freshness", "Prediction required-field and key integrity",
            "PASS" if nulls == 0 and dup == 0 and invalid == 0 else "FAIL",
            f"rows={len(p)}; missing_required_cells={nulls}; duplicate_keys={dup}; invalid_rows={invalid}",
            "data/predictions.csv")
        # Prediction history is intentionally retained; validate its latest session separately.
        latest = p["prediction_date"].max()
        latest_rows = p.loc[p["prediction_date"].eq(latest)]
        add(rows, "Live data quality and freshness", "Latest prediction session coverage",
            "PASS" if not latest_rows.empty and latest_rows["target_date"].gt(latest).all() else "FAIL",
            f"latest_prediction_date={latest.date() if pd.notna(latest) else 'unknown'}; rows={len(latest_rows)}; all_target_dates_after_prediction={bool(latest_rows['target_date'].gt(latest).all()) if not latest_rows.empty else False}",
            "data/predictions.csv")


    candidates = _check_csv_nonempty(rows, "Prediction candidate history", "data/prediction_candidates_history.csv",
        {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "rank", "score"}, 1)
    if candidates is not None:
        ch = candidates.copy()
        keys = ["prediction_date", "target_date", "symbol"]
        nulls = int(ch[keys + ["base_close", "predicted_close", "rank", "score"]].isna().sum().sum())
        dup = int(ch.duplicated(keys).sum())
        add(rows, "Live data quality and freshness", "Candidate-history completeness and uniqueness",
            "PASS" if nulls == 0 and dup == 0 else "FAIL",
            f"rows={len(ch)}; required_null_cells={nulls}; duplicate_candidate_keys={dup}",
            "data/prediction_candidates_history.csv")
        if (ROOT / "data/predictions.csv").exists():
            pred = _read_csv(ROOT / "data/predictions.csv")
            if set(keys).issubset(pred.columns) and set(keys).issubset(ch.columns) and not pred.empty:
                pred["prediction_date"] = pd.to_datetime(pred["prediction_date"], errors="coerce")
                ch["prediction_date"] = pd.to_datetime(ch["prediction_date"], errors="coerce")
                latest_pred = pred["prediction_date"].max()
                expected = pred.loc[pred["prediction_date"].eq(latest_pred), keys].drop_duplicates()
                actual = ch[keys].drop_duplicates()
                for frame in (expected, actual):
                    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
                    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
                missing = expected.merge(actual, on=keys, how="left", indicator=True)
                missing_count = int(missing["_merge"].eq("left_only").sum())
                add(rows, "Live data quality and freshness", "Latest predictions covered by candidate history",
                    "PASS" if missing_count == 0 else "FAIL",
                    f"latest_prediction_date={latest_pred.date() if pd.notna(latest_pred) else 'unknown'}; current_predictions={len(expected)}; missing_candidate_history={missing_count}",
                    "data/predictions.csv + data/prediction_candidates_history.csv")

    evaluations = _check_csv_nonempty(rows, "Prediction evaluation outcomes", "data/evaluations.csv",
        {"prediction_date", "target_date", "symbol", "predicted_close", "actual_close", "close_error"}, 1)
    if evaluations is not None:
        ev = evaluations.copy()
        for c in ["prediction_date", "target_date"]:
            ev[c] = pd.to_datetime(ev[c], errors="coerce")
        numeric = ["predicted_close", "actual_close", "close_error"]
        for c in numeric:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
        nulls = int(ev[["prediction_date", "target_date", "symbol"] + numeric].isna().sum().sum())
        dup = int(ev.duplicated(["prediction_date", "target_date", "symbol"]).sum())
        future_targets = int((ev["target_date"] > pd.Timestamp.now().normalize()).sum())
        add(rows, "Live data quality and freshness", "Evaluation outcome completeness",
            "PASS" if nulls == 0 and dup == 0 and future_targets == 0 else "FAIL",
            f"rows={len(ev)}; missing_required_cells={nulls}; duplicate_keys={dup}; future_unrealized_targets={future_targets}",
            "data/evaluations.csv")

    for rel, label, version in [
        ("data/paper_trades.csv", "V1 paper-trading ledger", 1),
        ("data/paper_trades_v2.csv", "V2 shadow-trading ledger", 2),
    ]:
        required = {"symbol", "target_date", "signal", "quantity", "planned_capital"} if version == 2 else {"symbol", "target_date", "signal"}
        ledger = _check_csv_nonempty(rows, label, rel, required, 1)
        if ledger is None:
            continue
        signal = ledger["signal"].fillna("").astype(str).str.upper().str.strip()
        buys = ledger.loc[signal.eq("BUY")].copy()
        skips = ledger.loc[signal.eq("SKIP")].copy()
        missing_buy = 0
        if version == 2:
            for col in ["quantity", "planned_capital"]:
                buys[col] = pd.to_numeric(buys[col], errors="coerce")
                missing_buy += int((~np.isfinite(buys[col])).sum())
                if col in buys:
                    missing_buy += int((buys[col] <= 0).sum())
            reason_missing = int(skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip().eq("").sum())
        else:
            for col in ["entry_price", "exit_price", "profit_loss", "return_pct"]:
                if col in buys:
                    buys[col] = pd.to_numeric(buys[col], errors="coerce")
                    missing_buy += int((~np.isfinite(buys[col])).sum())
            reason_missing = 0
        add(rows, "Live data quality and freshness", f"{label} BUY/skip field integrity",
            "PASS" if missing_buy == 0 and reason_missing == 0 else "REVIEW",
            f"rows={len(ledger)}; BUY_rows={len(buys)}; BUY_missing_or_invalid_fields={missing_buy}; SKIP_rows_without_reason={reason_missing}",
            rel)

    # The broad market intelligence output is a persisted, date-stamped input to daily selection.
    intel = _check_csv_nonempty(rows, "Market/index intelligence output", "data/index_intelligence_summary.csv",
        {"index", "as_of", "regime", "direction", "data_quality", "index_data_available"}, 1)
    if intel is not None:
        ix = intel.copy()
        ix["as_of"] = pd.to_datetime(ix["as_of"], errors="coerce")
        latest = ix["as_of"].max()
        unavailable = int(ix["index_data_available"].astype(str).str.lower().isin({"false", "0", "no"}).sum())
        bad_quality = int((~ix["data_quality"].astype(str).str.upper().isin({"HIGH", "PASS", "OK"})).sum())
        add(rows, "Live data quality and freshness", "Market intelligence freshness and source status",
            "PASS" if pd.notna(latest) and unavailable == 0 and bad_quality == 0 else "REVIEW",
            f"rows={len(ix)}; latest_as_of={latest.date() if pd.notna(latest) else 'unknown'}; unavailable_index_rows={unavailable}; non_high_quality_rows={bad_quality}",
            "data/index_intelligence_summary.csv")

    # Confirm the production and weekend workflows actually schedule the refreshers
    # and persist the outputs. This catches a data module that exists but is never run.
    daily_path = ROOT / ".github/workflows/daily.yml"
    weekend_path = ROOT / ".github/workflows/weekend-data-audit.yml"
    wiring = [
        (daily_path, "Daily official calendar refresh", "src.market_calendar --check-today"),
        (daily_path, "Daily index OHLCV and constituent refresh", "src.index_constituent_intelligence"),
        (weekend_path, "Weekend official calendar refresh", "src.market_calendar"),
        (weekend_path, "Weekend stock-history quality refresh", "src.weekend_data_quality_audit"),
        (weekend_path, "Weekend index data refresh", "src.index_constituent_intelligence"),
    ]
    for path, label, needle in wiring:
        present = path.exists() and needle in path.read_text(encoding="utf-8")
        add(rows, "Refresh automation wiring", label, "PASS" if present else "FAIL",
            "Refresh command is wired into workflow" if present else f"Refresh command missing from {path.relative_to(ROOT)}",
            str(path.relative_to(ROOT)))

    # Explicitly surface whether the index features are actually consumed by the production model.
    pipeline_text = (ROOT / "src/pipeline.py").read_text(encoding="utf-8") if (ROOT / "src/pipeline.py").exists() else ""
    consumed = "index_constituent_features" in pipeline_text
    add(rows, "Live data quality and freshness", "Index features connected to production prediction pipeline",
        "PASS" if consumed else "BLOCKED",
        "Production pipeline " + ("references index constituent feature data" if consumed else "does not yet load index_constituent_features.csv; collection alone does not mean model learns from it"),
        "src/pipeline.py")

def run_audit(output_dir: Path):
    rows = []
    audit_live_data(rows)
    for name, rel in MODULES.items():
        exists = (ROOT / rel).is_file()
        add(rows, "Implementation structure", name, "PASS" if exists else "FAIL",
            "Source module exists" if exists else "Required source module is missing", rel)

    for name, (rel, required_columns) in ARTIFACTS.items():
        path = ROOT / rel
        if not path.is_file():
            add(rows, "Data and evidence artifacts", name, "BLOCKED",
                "Artifact has not been produced or is not present; dependent validation cannot be confirmed", rel)
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = inspect_json(path)
                status = payload.get("status") if isinstance(payload, dict) else None
                detail = f"Valid JSON; reported status={status or 'not supplied'}"
                if status and str(status).upper() not in {"PASS", "OK", "HEALTHY", "COMPLETE", "SUFFICIENT"}:
                    detail += " (evidence exists but its own status is not a clean pass)"
                    result = "REVIEW"
                else:
                    result = "PASS"
            else:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                missing = sorted(required_columns - set(header)) if required_columns else []
                if missing:
                    result, detail = "FAIL", f"Missing required columns: {missing}"
                else:
                    result, detail = "PASS", f"Readable CSV; {len(header)} columns; header validated"
            add(rows, "Data and evidence artifacts", name, result, detail, rel)
        except Exception as exc:
            add(rows, "Data and evidence artifacts", name, "FAIL", f"Cannot parse artifact: {type(exc).__name__}: {exc}", rel)

    present = [p for p in PHASE_EVIDENCE if (ROOT / p).is_file()]
    missing = [p for p in PHASE_EVIDENCE if not (ROOT / p).is_file()]
    add(rows, "Roadmap evidence coverage", "Phase evidence summaries",
        "PASS" if not missing else "REVIEW",
        f"{len(present)}/{len(PHASE_EVIDENCE)} expected phase evidence summaries present; "
        f"missing: {', '.join(missing) if missing else 'none'}",
        "data/phase*_summary.json")

    for rel in ["README.md", ".github/workflows/daily.yml",
                ".github/workflows/weekend-data-audit.yml",
                ".github/workflows/pipeline-integrity-ci.yml"]:
        exists = (ROOT / rel).is_file()
        add(rows, "Operations and documentation", rel, "PASS" if exists else "FAIL",
            "Present" if exists else "Missing", rel)

    # Report important evidence gates as evidence, not as a binary code-health result.
    for rel, label in [
        ("data/phase114_promotion_readiness_summary.json", "Production promotion gate"),
        ("data/phase107_pnl_reconciliation_summary.json", "Paper-trade P&L reconciliation"),
        ("data/historical_coverage_audit.json", "Historical data coverage"),
    ]:
        path = ROOT / rel
        if not path.exists():
            add(rows, "Model readiness evidence", label, "BLOCKED", "No report available", rel)
            continue
        try:
            payload = inspect_json(path)
            status = str(payload.get("status", "STATUS_NOT_REPORTED")).upper()
            detail = f"Current evidence report status: {status}"
            if status in {"PASS", "SUFFICIENT", "READY", "COMPLETE"}:
                result = "PASS"
            elif status in {"FAIL", "BLOCKED", "INSUFFICIENT_EVIDENCE"}:
                result = "BLOCKED"
            else:
                result = "REVIEW"
            unmet = payload.get("unmet_gates", [])
            if unmet:
                detail += f"; unmet gates: {', '.join(map(str, unmet))}"
            add(rows, "Model readiness evidence", label, result, detail, rel)
        except Exception as exc:
            add(rows, "Model readiness evidence", label, "FAIL", f"Invalid report: {exc}", rel)

    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ["PASS", "FAIL", "BLOCKED", "REVIEW"]}
    total = len(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "repository structure and persisted evidence; not a profitability certification",
        "counts": counts, "total_checks": total, "checks": rows,
    }
    (output_dir / "model_progress_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Stock Picker — Model Progress Audit",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "> This audit separates code structure, data artifacts and readiness evidence. "
        "A present file is not proof of model skill, and BLOCKED/REVIEW are real findings, not workflow failures.",
        "",
        "## Summary",
        "",
        f"- Total checks: **{total}**",
        f"- PASS: **{counts['PASS']}**",
        f"- FAIL: **{counts['FAIL']}**",
        f"- BLOCKED: **{counts['BLOCKED']}**",
        f"- REVIEW: **{counts['REVIEW']}**",
        "",
        "## Detailed checks",
        "",
        "| Group | Check | Status | Detail | Evidence |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join(clean(r[k]) for k in ["group", "check", "status", "detail", "evidence"]) + " |")
    (output_dir / "model_progress_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"Full report: {output_dir / 'model_progress_audit.md'}")
    return 0 if counts["FAIL"] == 0 else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="audit/model_progress")
    args = parser.parse_args()
    raise SystemExit(run_audit(ROOT / args.output_dir))
