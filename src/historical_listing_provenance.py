"""Read-only audit of pre-NSE-listing history and exchange-provenance evidence.

The OHLCV store currently records Yahoo Finance-normalised bars, not the
exchange that originated each historical observation. Company/exchange events
can explain why pre-listing history may be legitimate, but they do not prove
the source exchange of an individual bar. This module deliberately preserves
that distinction and never edits or deletes OHLCV rows.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REGISTRY = "data/historical_listing_provenance.csv"
HISTORY = "data/ohlcv.csv"
LISTINGS = "data/nse_listing_master.csv"
REGISTRY_COLUMNS = {
    "symbol", "event_type", "predecessor_symbol", "evidence_start_date",
    "evidence_end_date", "evidence_exchange", "evidence_url",
    "evidence_note", "classification_status",
}


def audit_historical_listing_provenance(root: Path, history_frame: pd.DataFrame | None = None, listing_frame: pd.DataFrame | None = None) -> dict:
    """Compare pre-listing bars with verified event evidence without attribution guesses."""
    root = Path(root)
    registry_path = root / REGISTRY
    history_path = root / HISTORY
    listing_path = root / LISTINGS
    evidence = f"{REGISTRY} + {HISTORY} + {LISTINGS}"

    if not registry_path.is_file():
        return {"status": "BLOCKED", "detail": "Historical listing provenance registry is missing",
                "evidence": evidence}
    if not history_path.is_file() or not listing_path.is_file():
        missing = [str(p.relative_to(root)) for p in (history_path, listing_path) if not p.is_file()]
        return {"status": "BLOCKED", "detail": f"Required source artifacts missing: {missing}",
                "evidence": evidence}

    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    missing_columns = sorted(REGISTRY_COLUMNS - set(registry.columns))
    if missing_columns:
        return {"status": "FAIL", "detail": f"Provenance registry missing columns: {missing_columns}",
                "evidence": evidence}
    registry["symbol"] = registry["symbol"].astype(str).str.upper().str.strip()
    if registry["symbol"].eq("").any() or registry["symbol"].duplicated().any():
        return {"status": "FAIL", "detail": "Provenance registry has blank or duplicate symbols",
                "evidence": evidence}

    listings = (listing_frame.copy() if listing_frame is not None else pd.read_csv(listing_path, dtype=str)).fillna("")
    if not {"symbol", "listing_date"}.issubset(listings.columns):
        return {"status": "FAIL", "detail": "NSE listing master lacks symbol/listing_date",
                "evidence": evidence}
    listings["symbol"] = listings["symbol"].astype(str).str.upper().str.strip()
    listings["listing_date"] = pd.to_datetime(listings["listing_date"], format="mixed", errors="coerce", dayfirst=True)
    listing_map = (listings.dropna(subset=["listing_date"])
                   .drop_duplicates("symbol", keep="last")
                   .set_index("symbol")["listing_date"].to_dict())

    history_columns = set(history_frame.columns) if history_frame is not None else set(pd.read_csv(history_path, nrows=0).columns)
    bars = history_frame[["date", "symbol"]].copy() if history_frame is not None else pd.read_csv(history_path, usecols=["date", "symbol"])
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
    bars["symbol"] = bars["symbol"].astype(str).str.upper().str.strip()
    bars = bars.dropna(subset=["date"])
    details = []
    total_pre_listing = 0
    total_inside_event_windows = 0
    unresolved_symbols = []
    for row in registry.to_dict("records"):
        symbol = row["symbol"]
        listing_date = listing_map.get(symbol)
        stock = bars.loc[bars["symbol"].eq(symbol)].copy()
        if listing_date is None:
            unresolved_symbols.append(symbol)
            details.append(f"{symbol}:listing_date_unavailable")
            continue
        before = stock.loc[stock["date"].lt(listing_date)]
        count = int(len(before))
        total_pre_listing += count
        first = before["date"].min()
        last = before["date"].max()
        start = pd.to_datetime(row.get("evidence_start_date") or None, errors="coerce")
        end = pd.to_datetime(row.get("evidence_end_date") or None, errors="coerce")
        in_window = 0
        if pd.notna(start) and pd.notna(end):
            in_window = int(before["date"].between(start, end, inclusive="both").sum())
        total_inside_event_windows += in_window
        if count:
            details.append(
                f"{symbol}:pre_listing={count},first={first.date()},last={last.date()},"
                f"event_window_candidates={in_window},event={row.get('event_type') or 'unspecified'}"
            )
            # Event evidence is not row-level source attribution. Keep every bar
            # unresolved until an exchange/vendor record proves its origin.
            unresolved_symbols.append(symbol)

    source_fields = sorted(history_columns.intersection(
        {"exchange", "source_exchange", "provider", "source", "vendor_ticker"}
    ))
    source_attribution = "present" if source_fields else "absent"
    status = "REVIEW" if total_pre_listing or unresolved_symbols or not source_fields else "PASS"
    detail = (
        f"registry_symbols={len(registry)}; pre_listing_bars={total_pre_listing}; "
        f"bars_inside_documented_event_windows={total_inside_event_windows}; "
        f"symbols_requiring_row_level_attribution={sorted(set(unresolved_symbols))}; "
        f"OHLCV_exchange_or_provider_fields={source_attribution}{source_fields}; "
        f"per_symbol={details}; no OHLCV rows changed or deleted"
    )
    return {"status": status, "detail": detail, "evidence": evidence,
            "pre_listing_bars": total_pre_listing,
            "bars_inside_documented_event_windows": total_inside_event_windows,
            "unresolved_symbols": sorted(set(unresolved_symbols))}
