"""Verified NSE listing-date cache sourced from NSE's official quote-equity API."""
from __future__ import annotations
import os, tempfile, time, gzip
from io import BytesIO
from datetime import timedelta
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
    for fmt in ("%d-%b-%Y","%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%Y%m%d","%d%m%Y"):
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


SECURITY_MASTER_URL="https://nsearchives.nseindia.com/content/cm/NSE_CM_security_{date}.csv.gz"
SECURITY_MASTER_HEADERS={**HEADERS,"Referer":"https://www.nseindia.com/all-reports","Host":"nsearchives.nseindia.com"}


def _parse_mii_listing_date(raw):
    value=str(raw or "").strip()
    if not value or value in {"0","0.0","00000000"}:
        raise ValueError("NSE security master listing date is empty")
    # The official NSE master specification stores LONG date/time values as
    # seconds from 1980-01-01; accept explicit YYYYMMDD/date strings too.
    try:
        numeric=int(float(value))
        if 19000101 <= numeric <= 21001231 and len(value.split(".")[0]) == 8:
            return parse_date(str(numeric))
        if 0 < numeric < 2_000_000_000:
            return pd.Timestamp("1980-01-01") + pd.to_timedelta(numeric,unit="s")
    except (TypeError,ValueError,OverflowError):
        pass
    return parse_date(value)


def parse_security_master(payload: bytes, requested_symbols: set[str], source_url: str) -> list[dict]:
    """Extract verified listing dates from NSE's daily MII CM security master."""
    def build(symbol, company, isin, raw_date):
        try: day=_parse_mii_listing_date(raw_date)
        except Exception: return None
        return {"symbol":str(symbol).strip().upper(),"company_name":str(company or "").strip(),
            "isin":str(isin or "").strip().upper(),"listing_date":day.date().isoformat(),
            "listing_date_source":source_url,"verified_at_utc":datetime.now().astimezone().isoformat()}
    # Newer CSV-style MII exports may publish named columns.
    try:
        try: frame=pd.read_csv(BytesIO(payload),compression="gzip",low_memory=False)
        except Exception: frame=pd.read_csv(BytesIO(payload),low_memory=False)
        def key(value):
            return "".join(ch for ch in str(value).lower() if ch.isalnum())
        columns={key(c):c for c in frame.columns}
        symbol_col=next((columns[k] for k in ("symbol","tradingsymbol","securitysymbol") if k in columns),None)
        listing_col=next((columns[k] for k in ("listingdate","dateoflisting") if k in columns),None)
        isin_col=next((columns[k] for k in ("isin","isinnumber","isinno") if k in columns),None)
        company_col=next((columns[k] for k in ("companyname","securityname","issuername","name") if k in columns),None)
        if symbol_col and listing_col:
            frame[symbol_col]=frame[symbol_col].astype(str).str.upper().str.strip()
            frame=frame[frame[symbol_col].isin(requested_symbols)]
            found=[]
            for _,row in frame.iterrows():
                item=build(row[symbol_col],row[company_col] if company_col else "",
                    row[isin_col] if isin_col else "",row[listing_col])
                if item: found.append(item)
            if found: return found
    except Exception:
        pass
    # NSE's published master specification also defines the legacy NEATCM
    # pipe-delimited record layout. Header record is followed by data records.
    raw_bytes=payload
    if raw_bytes[:2] == b"\\x1f\\x8b":
        raw_bytes=gzip.decompress(raw_bytes)
    lines=raw_bytes.decode("utf-8",errors="replace").splitlines()
    found=[]
    for line in lines:
        fields=line.strip().split("|")
        if len(fields)<54 or fields[0].strip().upper().startswith("NEATCM"):
            continue
        symbol=fields[1].strip().upper()
        if symbol not in requested_symbols:
            continue
        # NEATCM CM layout: Name=field 21, Listing Date=field 27,
        # ISIN Number=field 53 (zero-based positions).
        item=build(symbol,fields[21],fields[53],fields[27])
        if item: found.append(item)
    if not found:
        raise ValueError("NSE MII master has no parseable requested symbols/listing dates; expected named CSV columns or documented NEATCM pipe layout")
    return found


def refresh_from_security_master(client, unresolved: set[str], max_calendar_days: int = 10) -> tuple[list[dict],dict]:
    if not unresolved:
        return [],{"status":"SKIPPED","reason":"No unresolved symbols"}
    from datetime import datetime as dt
    today=dt.now().date()
    attempts=[]; errors=[]
    for offset in range(max_calendar_days):
        candidate=today-timedelta(days=offset)
        url=SECURITY_MASTER_URL.format(date=candidate.strftime("%d%m%Y"))
        try:
            response=client.get(url,timeout=25,headers=SECURITY_MASTER_HEADERS)
            response.raise_for_status()
            found=parse_security_master(response.content,unresolved,url)
            attempts.append(candidate.isoformat())
            if found:
                return found,{"status":"PASS","source":url,"symbols_found":len(found),"attempted_dates":attempts}
            errors.append({"date":candidate.isoformat(),"error":"Valid file but no requested listing rows parsed"})
        except Exception as exc:
            errors.append({"date":candidate.isoformat(),"error":str(exc)})
        attempts.append(candidate.isoformat())
    return [],{"status":"SOURCE_UNAVAILABLE","attempted_dates":attempts,"errors":errors[-5:]}

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
    home_error=""
    try: client.get(HOME,timeout=20)
    except Exception as exc:
        home_error=str(exc)
    found=[]
    failures={s:f"Quote API session warm-up failed: {home_error}" for s in todo} if home_error else {}
    if home_error:
        todo=[]
    consecutive_systemic_failures=0
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
                      ("404" in lower_error and "nseindia" in lower_error) or
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
    unresolved_set=set(requested)-cached
    master_found, master_audit=refresh_from_security_master(client,unresolved_set)
    if master_found:
        merged=pd.concat([merged,pd.DataFrame(master_found)],ignore_index=True).drop_duplicates("symbol",keep="last").reindex(columns=COLS)
    cached=set(merged.loc[merged.listing_date.astype(str).str.strip().ne(""),"symbol"])
    unresolved=sorted(set(requested)-cached)
    # Persist a schema-bearing file even when NSE is unavailable. An empty file
    # is explicitly reported as unavailable; it is never mistaken for coverage.
    fd,tmp=tempfile.mkstemp(prefix="nse-listings-",suffix=".csv",dir=DATA)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f: merged.reindex(columns=COLS).to_csv(f,index=False)
        os.replace(tmp,FILE)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return {"status":"PASS" if not unresolved else ("PARTIAL" if cached & set(requested) else "SOURCE_UNAVAILABLE"),
            "symbols_requested":len(requested),"symbols_cached":len(cached & set(requested)),
            "symbols_fetched":len(found),"symbols_from_security_master":len(master_found),
            "symbols_unresolved":unresolved,"lookup_failures":failures,
            "security_master_fallback":master_audit,
            "source":"NSE official quote-equity API and daily MII CM security master",
            "cache_file":"data/nse_listing_master.csv"}
