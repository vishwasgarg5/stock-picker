from __future__ import annotations

"""Fail-closed point-in-time feature and security-universe evidence helpers."""
import re
from urllib.parse import urlparse
import pandas as pd

_SHA=re.compile(r"^[0-9a-f]{64}$")


def validate_feature_provenance(features: pd.DataFrame) -> pd.DataFrame:
    required={"security_id","feature_name","value","published_at","first_seen_at","source_url","source_sha256"}
    missing=required.difference(features.columns)
    if missing: raise ValueError(f"feature table missing columns: {sorted(missing)}")
    x=features.copy()
    for col in ("security_id","feature_name","source_url","source_sha256"):
        x[col]=x[col].astype("string").str.strip()
        if x[col].isna().any() or x[col].eq("").any(): raise ValueError(f"{col} contains blank values")
    if x["security_id"].str.upper().ne(x["security_id"]).any(): raise ValueError("security_id must be canonical uppercase")
    if not x["source_sha256"].map(lambda v: bool(_SHA.fullmatch(str(v)))).all(): raise ValueError("source_sha256 must be lowercase SHA-256")
    for url in x["source_url"]:
        p=urlparse(str(url))
        if p.scheme!="https" or not p.netloc: raise ValueError("source_url must be absolute HTTPS")
    for col in ("published_at","first_seen_at"):
        x[col]=pd.to_datetime(x[col],errors="coerce",utc=True)
        if x[col].isna().any(): raise ValueError(f"{col} contains invalid timestamps")
    x["available_at"]=x[["published_at","first_seen_at"]].max(axis=1)
    if x["value"].isna().any(): raise ValueError("feature values cannot be missing")
    key=["security_id","feature_name","available_at","source_sha256"]
    if x.duplicated(key).any(): raise ValueError("duplicate feature provenance records")
    return x.sort_values(["security_id","feature_name","available_at"],kind="mergesort").reset_index(drop=True)


def point_in_time_feature_snapshot(features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    required={"signal_id","security_id","signal_timestamp"}
    missing=required.difference(signals.columns)
    if missing: raise ValueError(f"signal table missing columns: {sorted(missing)}")
    f=validate_feature_provenance(features)
    s=signals.copy()
    s["security_id"]=s["security_id"].astype("string").str.strip()
    s["signal_timestamp"]=pd.to_datetime(s["signal_timestamp"],errors="coerce",utc=True)
    if s["signal_timestamp"].isna().any() or s["signal_id"].isna().any(): raise ValueError("signal identity/timestamp invalid")
    if s.duplicated(["signal_id","security_id"]).any(): raise ValueError("duplicate signal_id/security_id")
    rows=[]
    for signal in s.to_dict("records"):
        eligible=f[(f["security_id"]==signal["security_id"]) & (f["available_at"]<=signal["signal_timestamp"])]
        latest=eligible.sort_values("available_at",kind="mergesort").groupby("feature_name",sort=True).tail(1)
        for item in latest.to_dict("records"):
            rows.append({"signal_id":signal["signal_id"],"security_id":signal["security_id"],
                         "signal_timestamp":signal["signal_timestamp"],"feature_name":item["feature_name"],
                         "value":item["value"],"available_at":item["available_at"],
                         "source_url":item["source_url"],"source_sha256":item["source_sha256"]})
    return pd.DataFrame(rows,columns=["signal_id","security_id","signal_timestamp","feature_name","value","available_at","source_url","source_sha256"])


def audit_security_master(history: pd.DataFrame) -> dict:
    """Audit effective-dated identity rows; never infer missing history."""
    required={"security_id","symbol","instrument_type","series","valid_from","valid_to","source_url","source_published_at"}
    missing=required.difference(history.columns)
    if missing: return {"status":"BLOCKED_SCHEMA","missing_columns":sorted(missing),"rows":len(history)}
    x=history.copy()
    for c in ("security_id","symbol","instrument_type","series","source_url"):
        x[c]=x[c].astype("string").str.strip()
    x["valid_from"]=pd.to_datetime(x["valid_from"],errors="coerce").dt.normalize()
    x["valid_to"]=pd.to_datetime(x["valid_to"],errors="coerce").dt.normalize()
    x["source_published_at"]=pd.to_datetime(x["source_published_at"],errors="coerce").dt.normalize()
    errors=[]
    for c in ("security_id","symbol","instrument_type","series","source_url"):
        if x[c].isna().any() or x[c].eq("").any(): errors.append(f"blank_{c}")
    if x["valid_from"].isna().any() or x["source_published_at"].isna().any(): errors.append("invalid_effective_or_publication_date")
    if (x["valid_to"].notna() & (x["valid_to"]<x["valid_from"])).any(): errors.append("valid_to_before_valid_from")
    if (x["source_published_at"]>x["valid_from"]).any(): errors.append("source_published_after_effective_date")
    for url in x["source_url"].dropna():
        parsed=urlparse(str(url))
        if parsed.scheme!="https" or not parsed.netloc:
            errors.append("source_url_not_https"); break
    for sid,g in x.sort_values(["security_id","valid_from"]).groupby("security_id"):
        previous_end=None
        for row in g.itertuples():
            if previous_end is not None and (pd.isna(previous_end) or row.valid_from<=previous_end):
                errors.append(f"overlapping_identity_intervals:{sid}"); break
            previous_end=row.valid_to
    return {"status":"PASS_SCHEMA_AND_INTERVALS" if not errors else "BLOCKED",
            "rows":int(len(x)),"security_count":int(x["security_id"].nunique()),
            "errors":sorted(set(errors)),
            "point_in_time_complete":False,
            "note":"A passing structural audit does not prove NSE source authenticity, inactive-security completeness, corporate-action continuity, or historical tradability."}
