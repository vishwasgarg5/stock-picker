"""Verified NSE listing-date cache sourced from NSE's official quote-equity API."""
from __future__ import annotations
import os, tempfile, time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
FILE=DATA/"nse_listing_master.csv"
HOME="https://www.nseindia.com/"
URL="https://www.nseindia.com/api/quote-equity?symbol={}"
HEADERS={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36","Accept":"application/json,text/plain,*/*","Referer":HOME}
COLS=["symbol","company_name","isin","listing_date","listing_date_source","verified_at_utc"]

def parse_date(value):
    raw=str(value or "").strip()
    for fmt in ("%d-%b-%Y","%Y-%m-%d","%d/%m/%Y","%d-%m-%Y"):
        try: return pd.Timestamp(datetime.strptime(raw,fmt).date())
        except ValueError: pass
    parsed=pd.to_datetime(raw,errors="coerce",dayfirst=True)
    if pd.isna(parsed): raise ValueError(f"Invalid NSE listing date: {raw}")
    return pd.Timestamp(parsed).normalize()

def normalise_quote(symbol,payload):
    info=payload.get("info") if isinstance(payload,dict) else None
    if not isinstance(info,dict): raise ValueError("NSE quote response missing info object")
    day=parse_date(info.get("listingDate") or info.get("listing_date"))
    isin=str(info.get("isin") or "").strip().upper()
    if isin and (len(isin)!=12 or not isin.startswith("IN")): raise ValueError(f"Invalid ISIN for {symbol}: {isin}")
    sym=symbol.strip().upper()
    return {"symbol":sym,"company_name":str(info.get("companyName") or "").strip(),"isin":isin,
            "listing_date":day.date().isoformat(),"listing_date_source":URL.format(quote(sym)),
            "verified_at_utc":datetime.now().astimezone().isoformat()}

def refresh_listing_master(symbols,session=None,delay_seconds=0.15,max_attempts=3):
    DATA.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(FILE,dtype=str).fillna("") if FILE.exists() else pd.DataFrame(columns=COLS)
    for col in COLS:
        if col not in old: old[col]=""
    old["symbol"]=old["symbol"].astype(str).str.upper().str.strip()
    old=old[old.symbol.ne("")].drop_duplicates("symbol",keep="last")
    requested=sorted({str(s).strip().upper() for s in symbols if str(s).strip() and not str(s).strip().upper().startswith("DUMMY")})
    known=set(old.loc[old.listing_date.astype(str).str.strip().ne(""),"symbol"])
    todo=[s for s in requested if s not in known]
    client=session or requests.Session()
    client.headers.update(HEADERS)
    try: client.get(HOME,timeout=20)
    except Exception as exc:
        return {"status":"SOURCE_UNAVAILABLE" if todo else "PASS","symbols_requested":len(requested),
                "symbols_cached":len(known & set(requested)),"symbols_fetched":0,"symbols_unresolved":todo,"error":str(exc)}
    found=[]; failures={}; consecutive_systemic_failures=0
    for i,symbol in enumerate(todo):
        error="NSE lookup failed"
        for attempt in range(max_attempts):
            try:
                response=client.get(URL.format(quote(symbol)),timeout=25); response.raise_for_status()
                found.append(normalise_quote(symbol,response.json())); error=""; break
            except Exception as exc:
                error=str(exc)
                if attempt+1<max_attempts: time.sleep(min(2**attempt,4))
        if error:
            failures[symbol]=error
            lower_error=error.lower()
            systemic=(any(token in lower_error for token in
                      ("403", "429", "timed out", "timeout", "connection",
                       "502", "503", "504", "server error")) or
                      ("404" in lower_error and "nseindia.in" in lower_error) or
                      "nse quote response missing info object" in lower_error)
            consecutive_systemic_failures = consecutive_systemic_failures + 1 if systemic else 0
            # Stop a long 500-symbol retry storm when NSE is rejecting the
            # entire session/network. Unresolved symbols remain explicit and
            # are retried on the next scheduled refresh.
            if consecutive_systemic_failures >= 10:
                for remaining in todo[i+1:]:
                    failures[remaining]="Skipped after 10 consecutive systemic NSE errors; will retry next refresh"
                break
        else:
            consecutive_systemic_failures=0
        if delay_seconds and i+1<len(todo): time.sleep(delay_seconds)
    merged=old
    if found:
        merged=pd.concat([old,pd.DataFrame(found)],ignore_index=True).drop_duplicates("symbol",keep="last").reindex(columns=COLS)
        fd,tmp=tempfile.mkstemp(prefix="nse-listings-",suffix=".csv",dir=DATA)
        try:
            with os.fdopen(fd,"w",encoding="utf-8",newline="") as f: merged.to_csv(f,index=False)
            os.replace(tmp,FILE)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    cached=set(merged.loc[merged.listing_date.astype(str).str.strip().ne(""),"symbol"])
    unresolved=sorted(set(requested)-cached)
    return {"status":"PASS" if not unresolved else ("PARTIAL" if cached & set(requested) else "SOURCE_UNAVAILABLE"),
            "symbols_requested":len(requested),"symbols_cached":len(cached & set(requested)),
            "symbols_fetched":len(found),"symbols_unresolved":unresolved,"lookup_failures":failures,
            "source":"NSE India official quote-equity API info.listingDate","cache_file":"data/nse_listing_master.csv"}
