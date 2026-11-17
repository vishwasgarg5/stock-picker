from __future__ import annotations

"""Immutable source-manifest checks. Hash consistency is not source authentication."""
import hashlib
import re
from urllib.parse import urlparse
import pandas as pd

_SHA=re.compile(r"^[0-9a-f]{64}$")
_REQUIRED={"source_id","source_url","source_sha256","retrieved_at"}


def validate_source_manifest(records: pd.DataFrame) -> pd.DataFrame:
    missing=_REQUIRED.difference(records.columns)
    if missing: raise ValueError(f"source manifest missing columns: {sorted(missing)}")
    x=records.copy()
    for col in ("source_id","source_url","source_sha256"):
        x[col]=x[col].astype("string").str.strip()
        if x[col].isna().any() or x[col].eq("").any(): raise ValueError(f"{col} contains blank values")
    if x.source_id.duplicated().any(): raise ValueError("source_id must be unique")
    if not x.source_sha256.map(lambda v: bool(_SHA.fullmatch(str(v)))).all(): raise ValueError("invalid SHA-256 digest")
    for value in x.source_url:
        u=urlparse(str(value))
        if u.scheme!="https" or not u.netloc: raise ValueError("source_url must be HTTPS")
    x["retrieved_at"]=pd.to_datetime(x["retrieved_at"],errors="coerce",utc=True)
    if x.retrieved_at.isna().any(): raise ValueError("retrieved_at must contain valid timestamps")
    return x


def verify_source_payloads(records: pd.DataFrame, payloads: dict[str,bytes]) -> dict:
    manifest=validate_source_manifest(records)
    expected=set(manifest.source_id.astype(str)); supplied=set(payloads)
    missing=sorted(expected-supplied); unexpected=sorted(supplied-expected); mismatches=[]
    for row in manifest.to_dict("records"):
        sid=str(row["source_id"])
        if sid in payloads and hashlib.sha256(payloads[sid]).hexdigest()!=str(row["source_sha256"]):
            mismatches.append(sid)
    return {"passed":not missing and not unexpected and not mismatches,
            "verified_count":len(expected)-len(missing)-len(mismatches),
            "missing_payloads":missing,"unexpected_payloads":unexpected,
            "hash_mismatches":sorted(mismatches),
            "authenticity_verified":False,"historical_validity_verified":False}
