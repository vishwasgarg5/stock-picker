from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNIVERSE_FILE = DATA / "universe.csv"

# Official NSE archive endpoint for the current Nifty Midcap 150 constituent list.
NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv"


def fetch_universe() -> pd.DataFrame:
    response = requests.get(
        NSE_URL,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"},
        timeout=30,
    )
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

    out = pd.DataFrame({"symbol": df[symbol_col].astype(str).str.strip()})
    out["company_name"] = df[name_col].astype(str).str.strip() if name_col else out["symbol"]
    out = out[out["symbol"].ne("") & out["symbol"].ne("nan")]
    out = out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

    if len(out) < 140 or len(out) > 160:
        raise ValueError(f"Expected about 150 constituents, received {len(out)}")
    return out


def update_universe() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    df = fetch_universe()
    df.to_csv(UNIVERSE_FILE, index=False)
    print(f"Updated Nifty Midcap 150 universe: {len(df)} stocks")
    return df


if __name__ == "__main__":
    update_universe()
