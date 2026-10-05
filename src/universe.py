from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNIVERSE_FILE = DATA / "universe.csv"

# Official NSE Nifty 500 constituent file.
NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_HOME = "https://www.nseindia.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Referer": NSE_HOME,
    "Connection": "keep-alive",
}

MIN_CONSTITUENTS = 450
MAX_CONSTITUENTS = 550

NEW_LISTINGS_FILE = DATA / "new_listings.csv"
NEW_LISTINGS_HISTORY_FILE = DATA / "new_listings_history.csv"
NEW_LISTINGS_URL = "https://www.nseindia.com/api/new-listings-today"

NEW_LISTING_MONITOR_DAYS = 4
NEW_LISTING_LIMITED_DAYS = 19
NEW_LISTING_REDUCED_CONFIDENCE_DAYS = 59


def _normalise_new_listing_rows(payload: object) -> pd.DataFrame:
    """Normalise NSE's daily New IPO/Listing feed."""
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or payload.get("records") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    if not rows:
        return pd.DataFrame(columns=["symbol", "company_name", "listing_date", "listing_type"])

    frame = pd.DataFrame(rows)
    rename = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"symbol", "sym"}:
            rename[col] = "symbol"
        elif key in {"company_name", "company", "name", "companyname"}:
            rename[col] = "company_name"
        elif key in {"listing_date", "date", "list_date", "listingdate"}:
            rename[col] = "listing_date"
        elif key in {"security_type", "type", "series"}:
            rename[col] = "listing_type"
    frame = frame.rename(columns=rename)
    if "symbol" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "company_name", "listing_date", "listing_type"])
    for col in ["company_name", "listing_date", "listing_type"]:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame = frame[frame["symbol"].ne("") & frame["symbol"].ne("NAN")]
    frame["company_name"] = frame["company_name"].fillna(frame["symbol"]).astype(str).str.strip()
    frame["listing_date"] = pd.to_datetime(frame["listing_date"], errors="coerce", dayfirst=True)
    frame["listing_type"] = frame["listing_type"].fillna("NEW_LISTING").astype(str).str.strip()
    frame.loc[frame["listing_date"].isna(), "listing_date"] = pd.Timestamp.now().normalize()
    return frame[["symbol", "company_name", "listing_date", "listing_type"]].drop_duplicates("symbol")


def fetch_new_listings() -> pd.DataFrame:
    """Fetch stocks shown by NSE in its New IPO/Listing feed for today."""
    session = requests.Session()
    session.headers.update(HEADERS)
    for attempt in range(3):
        try:
            session.get(NSE_HOME, timeout=20)
            time.sleep(0.5)
            response = session.get(NEW_LISTINGS_URL, timeout=30)
            response.raise_for_status()
            return _normalise_new_listing_rows(response.json())
        except Exception as exc:
            if attempt == 2:
                print(f"New-listing feed unavailable after 3 attempts: {exc}")
            else:
                time.sleep(2 ** attempt)
    return pd.DataFrame(columns=["symbol", "company_name", "listing_date", "listing_type"])


def update_new_listings(core_universe: pd.DataFrame) -> pd.DataFrame:
    """Maintain a persistent recent-listing watchlist alongside Nifty 500."""
    detected = fetch_new_listings()
    core_symbols = set(core_universe["symbol"].astype(str).str.upper().str.strip())

    if NEW_LISTINGS_HISTORY_FILE.exists():
        history = pd.read_csv(NEW_LISTINGS_HISTORY_FILE)
    else:
        history = pd.DataFrame(columns=["symbol", "company_name", "listing_date", "listing_type"])

    for col in ["symbol", "company_name", "listing_date", "listing_type"]:
        if col not in history.columns:
            history[col] = pd.NA
    history["symbol"] = history["symbol"].astype(str).str.upper().str.strip()
    history["listing_date"] = pd.to_datetime(history["listing_date"], errors="coerce")

    combined = pd.concat([history, detected], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
    combined["listing_date"] = pd.to_datetime(combined["listing_date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["symbol", "listing_date"]).drop_duplicates("symbol", keep="last")

    today = pd.Timestamp.now().normalize()
    combined["calendar_age"] = (today - combined["listing_date"]).dt.days
    combined = combined[combined["calendar_age"].between(0, 120)].copy()

    watch = combined[~combined["symbol"].isin(core_symbols)].copy()
    watch["status"] = pd.cut(
        watch["calendar_age"],
        bins=[-1, NEW_LISTING_MONITOR_DAYS, NEW_LISTING_LIMITED_DAYS, NEW_LISTING_REDUCED_CONFIDENCE_DAYS, 10_000],
        labels=["MONITOR", "LIMITED_DATA", "ML_CANDIDATE_REDUCED_CONFIDENCE", "NORMAL"],
    ).astype(str)
    watch = watch[["symbol", "company_name", "listing_date", "listing_type", "calendar_age", "status"]].sort_values(
        ["listing_date", "symbol"], ascending=[False, True]
    )

    combined.to_csv(NEW_LISTINGS_HISTORY_FILE, index=False)
    watch.to_csv(NEW_LISTINGS_FILE, index=False)
    print(f"New-listing watchlist: {len(watch)} non-Nifty-500 recent listings")
    if not detected.empty:
        print("Detected today:", ", ".join(detected["symbol"].tolist()))
    return watch


def get_new_listing_symbols() -> list[str]:
    if not NEW_LISTINGS_FILE.exists():
        return []
    df = pd.read_csv(NEW_LISTINGS_FILE)
    if df.empty or "symbol" not in df.columns:
        return []
    return df["symbol"].dropna().astype(str).str.upper().str.strip().unique().tolist()




def fetch_universe() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            session.get(NSE_HOME, timeout=20)
            time.sleep(1)
            response = session.get(NSE_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(BytesIO(response.content))
            df.columns = [str(c).strip() for c in df.columns]

            symbol_col = next((c for c in df.columns if c.lower() == "symbol"), None)
            name_col = next(
                (c for c in df.columns if c.lower() in {"company name", "company_name", "companyname"}),
                None,
            )
            if not symbol_col:
                raise ValueError(f"NSE constituent file has no Symbol column: {df.columns.tolist()}")

            out = pd.DataFrame({"symbol": df[symbol_col].astype(str).str.strip().str.upper()})
            out["company_name"] = df[name_col].astype(str).str.strip() if name_col else out["symbol"]
            out = out[out["symbol"].ne("") & out["symbol"].ne("NAN")]
            out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

            if not MIN_CONSTITUENTS <= len(out) <= MAX_CONSTITUENTS:
                raise ValueError(
                    f"Expected Nifty 500 universe ({MIN_CONSTITUENTS}-{MAX_CONSTITUENTS} stocks), "
                    f"received {len(out)}"
                )
            return out
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Unable to download Nifty 500 from NSE after 3 attempts: {last_error}")


def update_universe() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        df = fetch_universe()
        df.to_csv(UNIVERSE_FILE, index=False)
        print(f"Updated Nifty 500 universe: {len(df)} stocks")
        update_new_listings(df)
        return df
    except Exception as exc:
        # Never destroy a previously valid universe because NSE is temporarily unavailable.
        if UNIVERSE_FILE.exists():
            existing = pd.read_csv(UNIVERSE_FILE)
            if "symbol" in existing.columns and MIN_CONSTITUENTS <= len(existing) <= MAX_CONSTITUENTS:
                print(f"NSE universe refresh failed ({exc}); using existing {len(existing)}-stock universe")
                return existing
        raise


if __name__ == "__main__":
    update_universe()
