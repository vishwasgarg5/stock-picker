from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNIVERSE_FILE = DATA / "universe.csv"

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv"
NSE_HOME = "https://www.nseindia.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Referer": NSE_HOME,
    "Connection": "keep-alive",
}


def fetch_universe() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            # Warm up the NSE session first; this materially reduces intermittent 403s.
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

            if not 140 <= len(out) <= 160:
                raise ValueError(f"Expected about 150 constituents, received {len(out)}")
            return out
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Unable to download Nifty Midcap 150 from NSE after 3 attempts: {last_error}")


def update_universe() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        df = fetch_universe()
        df.to_csv(UNIVERSE_FILE, index=False)
        print(f"Updated Nifty Midcap 150 universe: {len(df)} stocks")
        return df
    except Exception as exc:
        # Never destroy a previously valid universe because NSE is temporarily unavailable.
        if UNIVERSE_FILE.exists():
            existing = pd.read_csv(UNIVERSE_FILE)
            if "symbol" in existing.columns and len(existing) >= 140:
                print(f"NSE universe refresh failed ({exc}); using existing {len(existing)}-stock universe")
                return existing
        raise


if __name__ == "__main__":
    update_universe()
