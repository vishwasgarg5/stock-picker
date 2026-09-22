from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES = DATA / "prediction_candidates_history.csv"
HISTORY = DATA / "ohlcv.csv"
OUTPUT = DATA / "confidence_analysis.csv"
MIN_ROWS = 100
MIN_SESSIONS = 12

def run_confidence_analysis() -> pd.DataFrame:
    if not CANDIDATES.exists() or not HISTORY.exists():
        return pd.DataFrame()
    c = pd.read_csv(CANDIDATES)
    h = pd.read_csv(HISTORY, parse_dates=["date"])
    for col in ["prediction_date", "target_date"]:
        c[col] = pd.to_datetime(c[col], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    actual = h.rename(columns={"date":"target_date","open":"actual_open","high":"actual_high","low":"actual_low","close":"actual_close"})
    x = c.merge(actual[["symbol","target_date","actual_open","actual_high","actual_low","actual_close"]], on=["symbol","target_date"], how="inner")
    if x.empty:
        return pd.DataFrame()
    x["close_error_pct"] = (x["actual_close"]-x["predicted_close"]).abs()/x["actual_close"].abs()
    predicted_return = x["predicted_close"]/x["base_close"]-1
    actual_return = x["actual_close"]/x["base_close"]-1
    x["direction_correct"] = (np.sign(predicted_return)==np.sign(actual_return)).astype(int)
    x["mfe"] = x["actual_high"]/x["actual_open"]-1
    x["mae"] = x["actual_low"]/x["actual_open"]-1
    x["profitable_close"] = (x["actual_close"]/x["actual_open"]-1 > 0).astype(int)
    x["confidence_bucket"] = pd.cut(x["confidence_score"], bins=[-0.01,25,50,75,100.01], labels=["0-25","25-50","50-75","75-100"])
    out = x.groupby("confidence_bucket", observed=False).agg(
        rows=("symbol","count"), sessions=("target_date","nunique"),
        close_mape_pct=("close_error_pct", lambda s:s.mean()*100),
        direction_accuracy_pct=("direction_correct", lambda s:s.mean()*100),
        mfe_pct=("mfe", lambda s:s.mean()*100),
        mae_pct=("mae", lambda s:s.mean()*100),
        profitable_close_pct=("profitable_close", lambda s:s.mean()*100)
    ).reset_index()
    out["as_of"] = x["target_date"].max()
    out["minimum_rows_required"] = MIN_ROWS
    out["minimum_sessions_required"] = MIN_SESSIONS
    out["validation_status"] = np.where((out["rows"]>=MIN_ROWS)&(out["sessions"]>=MIN_SESSIONS),"validated_sample","collecting")
    out.to_csv(OUTPUT,index=False)
    return out

if __name__ == "__main__":
    result=run_confidence_analysis()
    print(f"Confidence analysis complete: {len(result)} buckets")
