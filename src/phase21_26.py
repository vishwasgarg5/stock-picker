from __future__ import annotations

"""Phases 21-26: validation, evidence, storage and drift guardrails.

All outputs are diagnostic. This module does not train, promote, or replace a
production model and never modifies prediction or trade decisions.
"""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUMMARY = DATA / "phase21_26_summary.json"
DIRECTION = DATA / "phase21_direction_calibration.csv"
LEAKAGE = DATA / "phase22_leakage_audit.csv"
AB = DATA / "phase23_matched_evidence.csv"
DRIFT = DATA / "phase25_drift_monitor.csv"
NEWS = DATA / "phase26_news_impact.csv"

def read_csv(name: str) -> pd.DataFrame:
    path = DATA / name
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()

def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def direction_calibration(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Report direction and error by confidence bucket; descriptive, not a filter."""
    cols = ["confidence_bucket", "rows", "sessions", "direction_accuracy_pct",
            "close_mape_pct", "baseline_mape_pct", "mape_lift_pct", "calibration_status"]
    if evaluations.empty:
        return pd.DataFrame(columns=cols)
    e = evaluations.copy()
    for col in ["close_direction_correct", "close_abs_pct_error", "baseline_close_abs_pct_error"]:
        if col not in e: e[col] = np.nan
    numeric(e, ["close_direction_correct", "close_abs_pct_error", "baseline_close_abs_pct_error"])
    e["close_mape_pct"] = e["close_abs_pct_error"] * 100
    e["baseline_mape_pct"] = e["baseline_close_abs_pct_error"] * 100
    if "prediction_date" in e:
        e["prediction_date"] = pd.to_datetime(e["prediction_date"], errors="coerce").dt.normalize()
    if not candidates.empty and {"prediction_date", "symbol"}.issubset(candidates.columns) and {"prediction_date", "symbol"}.issubset(e.columns):
        c = candidates.copy()
        c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
        c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
        e["symbol"] = e["symbol"].astype(str).str.upper().str.strip()
        conf_col = next((x for x in ["adjusted_confidence_score", "confidence_score", "confidence"] if x in c), None)
        if conf_col:
            c[conf_col] = pd.to_numeric(c[conf_col], errors="coerce")
            c = c[["prediction_date", "symbol", conf_col]].drop_duplicates(["prediction_date", "symbol"], keep="last")
            e = e.merge(c, on=["prediction_date", "symbol"], how="left")
            vals = e[conf_col]
            if vals.dropna().between(0, 1).mean() > 0.8:
                e["_confidence"] = vals * 100
            else:
                e["_confidence"] = vals
        else:
            e["_confidence"] = np.nan
    else:
        e["_confidence"] = pd.to_numeric(e.get("confidence_score", pd.Series(np.nan, index=e.index)), errors="coerce")
    e["confidence_bucket"] = pd.cut(e["_confidence"], [-np.inf, 40, 60, 80, np.inf],
                                    labels=["LOW_0_40", "MODERATE_40_60", "HIGH_60_80", "VERY_HIGH_80_PLUS"])
    e["confidence_bucket"] = e["confidence_bucket"].astype(object).where(e["_confidence"].notna(), "UNKNOWN")
    out = []
    for bucket, g in e.groupby("confidence_bucket", dropna=False, observed=False):
        d = g["close_direction_correct"].dropna()
        m = g["close_mape_pct"].dropna()
        b = g["baseline_mape_pct"].dropna()
        mm = float(m.mean()) if len(m) else np.nan
        bm = float(b.mean()) if len(b) else np.nan
        out.append({"confidence_bucket": str(bucket), "rows": len(g),
                    "sessions": int(g["prediction_date"].nunique()) if "prediction_date" in g else 0,
                    "direction_accuracy_pct": float(d.mean()*100) if len(d) else np.nan,
                    "close_mape_pct": mm, "baseline_mape_pct": bm,
                    "mape_lift_pct": float((bm-mm)/bm*100) if np.isfinite(mm) and np.isfinite(bm) and bm > 0 else np.nan,
                    "calibration_status": "EVIDENCE_ONLY" if len(g) >= 30 else "INSUFFICIENT_SAMPLE"})
    return pd.DataFrame(out, columns=cols)

def leakage_audit(evaluations: pd.DataFrame, candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Conservative date/duplicate/price integrity audit; never infers leakage from names alone."""
    findings = []
    def add(check, status, rows, detail):
        findings.append({"check": check, "status": status, "affected_rows": int(rows), "detail": detail})
    if evaluations.empty:
        add("evaluation_data", "BLOCKED", 0, "evaluations.csv missing or empty")
    else:
        e = evaluations.copy()
        p = pd.to_datetime(e.get("prediction_date"), errors="coerce")
        t = pd.to_datetime(e.get("target_date"), errors="coerce")
        bad = int((p.isna() | t.isna() | (p.dt.normalize() >= t.dt.normalize())).sum())
        add("prediction_before_target", "PASS" if bad == 0 else "WARN", bad,
            "prediction_date must be valid and earlier than target_date")
        if {"prediction_date", "symbol"}.issubset(e.columns):
            dup = int(e.assign(symbol=e["symbol"].astype(str).str.upper().str.strip(),
                               prediction_date=p.dt.normalize()).duplicated(["prediction_date", "symbol"]).sum())
            add("unique_prediction_key", "PASS" if dup == 0 else "WARN", dup, "duplicate prediction_date/symbol rows")
    if not candidates.empty and {"prediction_date", "symbol"}.issubset(candidates.columns):
        c = candidates.copy()
        dates = pd.to_datetime(c["prediction_date"], errors="coerce")
        dup = int(c.assign(symbol=c["symbol"].astype(str).str.upper().str.strip(),
                           prediction_date=dates.dt.normalize()).duplicated(["prediction_date", "symbol"]).sum())
        add("candidate_history_unique_key", "PASS" if dup == 0 else "WARN", dup, "duplicate candidate history keys")
        suspicious = [col for col in c.columns if re.search(r"(actual|target|future|next_day|forward_return|label)", col, re.I)]
        # Target/label columns may legitimately exist in a historical training table; flag for review only.
        add("candidate_future_named_columns", "REVIEW" if suspicious else "PASS", len(suspicious),
            "review columns: " + ", ".join(suspicious) if suspicious else "no obviously future/label-named candidate columns")
    if not history.empty and {"date", "symbol"}.issubset(history.columns):
        h = history.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        future = int((h["date"] > pd.Timestamp.now().normalize()).sum())
        dup = int(h.assign(symbol=h["symbol"].astype(str).str.upper().str.strip(),
                           date=h["date"].dt.normalize()).duplicated(["date", "symbol"]).sum())
        add("ohlcv_future_dates", "WARN" if future else "PASS", future, "OHLCV dates after current calendar date")
        add("ohlcv_duplicate_keys", "WARN" if dup else "PASS", dup, "duplicate date/symbol records")
    else:
        add("ohlcv_history", "REVIEW", 0, "history missing or lacks date/symbol columns")
    return pd.DataFrame(findings)

def matched_evidence(v1: pd.DataFrame, v2: pd.DataFrame, trades_v1: pd.DataFrame, trades_v2: pd.DataFrame) -> pd.DataFrame:
    """Compare only common sessions and expose evidence sufficiency; no promotion action."""
    def prepare(x, name):
        if x.empty or "target_date" not in x:
            return pd.DataFrame(columns=["target_date", name+"_pnl"])
        y = x.copy()
        y["target_date"] = pd.to_datetime(y["target_date"], errors="coerce").dt.normalize()
        col = next((c for c in ["daily_profit_loss", "pnl", "net_pnl"] if c in y), None)
        if col is None:
            y[name+"_pnl"] = np.nan
        else:
            y[name+"_pnl"] = pd.to_numeric(y[col], errors="coerce")
        return y.dropna(subset=["target_date"]).groupby("target_date", as_index=False)[name+"_pnl"].sum()
    a, b = prepare(v1, "v1"), prepare(v2, "v2")
    both = a.merge(b, on="target_date", how="inner")
    if both.empty:
        return pd.DataFrame([{"common_sessions": 0, "v1_net_pnl": np.nan, "v2_net_pnl": np.nan,
            "v2_minus_v1_pnl": np.nan, "v1_win_rate_pct": np.nan, "v2_win_rate_pct": np.nan,
            "v1_trade_count": len(trades_v1), "v2_trade_count": len(trades_v2),
            "evidence_gate": "COLLECTING_MATCHED_SESSIONS", "production_champion": "V1"}])
    both["v1_pnl"] = pd.to_numeric(both["v1_pnl"], errors="coerce")
    both["v2_pnl"] = pd.to_numeric(both["v2_pnl"], errors="coerce")
    return pd.DataFrame([{"common_sessions": len(both), "v1_net_pnl": float(both["v1_pnl"].sum()),
        "v2_net_pnl": float(both["v2_pnl"].sum()), "v2_minus_v1_pnl": float(both["v2_pnl"].sum()-both["v1_pnl"].sum()),
        "v1_win_rate_pct": float((both["v1_pnl"] > 0).mean()*100),
        "v2_win_rate_pct": float((both["v2_pnl"] > 0).mean()*100),
        "v1_trade_count": len(trades_v1), "v2_trade_count": len(trades_v2),
        "evidence_gate": "EVIDENCE_SUFFICIENT_FOR_REVIEW" if len(both) >= 20 and len(trades_v2) >= 50 else "COLLECTING_MATCHED_SESSIONS",
        "production_champion": "V1"}])

def drift_report(evaluations: pd.DataFrame) -> pd.DataFrame:
    cols = ["window", "sessions", "direction_accuracy_pct", "close_mape_pct", "baseline_mape_pct",
            "mape_lift_pct", "drift_status", "recommended_action"]
    if evaluations.empty or "target_date" not in evaluations:
        return pd.DataFrame([{"window":"recent_5_vs_prior","sessions":0,"drift_status":"INSUFFICIENT_DATA",
                              "recommended_action":"Keep V1; collect evaluation data"}], columns=cols)
    x = evaluations.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    numeric(x, ["close_direction_correct", "close_abs_pct_error", "baseline_close_abs_pct_error"])
    daily = x.dropna(subset=["target_date"]).groupby("target_date", as_index=False).agg(
        direction=("close_direction_correct", "mean"),
        mape=("close_abs_pct_error", "mean"), baseline=("baseline_close_abs_pct_error", "mean")).sort_values("target_date")
    if daily.empty:
        return pd.DataFrame([{"window":"recent_5_vs_prior","sessions":0,"drift_status":"INSUFFICIENT_DATA",
                              "recommended_action":"Keep V1; collect evaluation data"}], columns=cols)
    rows=[]
    for label, g in [("recent_5", daily.tail(5)), ("prior_5", daily.iloc[-10:-5])]:
        m=float(g["mape"].mean()*100) if g["mape"].notna().any() else np.nan
        b=float(g["baseline"].mean()*100) if g["baseline"].notna().any() else np.nan
        d=float(g["direction"].mean()*100) if g["direction"].notna().any() else np.nan
        lift=float((b-m)/b*100) if np.isfinite(m) and np.isfinite(b) and b>0 else np.nan
        rows.append({"window":label,"sessions":len(g),"direction_accuracy_pct":d,"close_mape_pct":m,
                     "baseline_mape_pct":b,"mape_lift_pct":lift,"drift_status":"MONITOR",
                     "recommended_action":"Review and consider rollback if sustained degradation; no automatic model change"})
    if len(daily) >= 10:
        recent, prior = rows[0], rows[1]
        degraded = ((np.isfinite(recent["mape_lift_pct"]) and np.isfinite(prior["mape_lift_pct"]) and recent["mape_lift_pct"] < prior["mape_lift_pct"]-5)
                    or (np.isfinite(recent["direction_accuracy_pct"]) and np.isfinite(prior["direction_accuracy_pct"]) and recent["direction_accuracy_pct"] < prior["direction_accuracy_pct"]-10))
        if degraded:
            rows[0]["drift_status"]="DEGRADATION_ALERT"
            rows[0]["recommended_action"]="Freeze challenger changes; inspect data/regime and review rollback; V1 stays champion"
    return pd.DataFrame(rows, columns=cols)

def news_impact_report(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Measure observed association between available news signals and later outcomes."""
    if evaluations.empty or candidates.empty or not {"prediction_date","symbol"}.issubset(evaluations.columns) or not {"prediction_date","symbol"}.issubset(candidates.columns):
        return pd.DataFrame([{"news_signal":"unavailable","bucket":"UNKNOWN","rows":0,"sessions":0,
            "direction_accuracy_pct":np.nan,"close_mape_pct":np.nan,"status":"INSUFFICIENT_DATA"}])
    e=evaluations.copy(); c=candidates.copy()
    for x in (e,c):
        x["prediction_date"]=pd.to_datetime(x["prediction_date"],errors="coerce").dt.normalize()
        x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    signals=[col for col in ["stock_news_score","sector_news_score"] if col in c]
    if not signals:
        return pd.DataFrame([{"news_signal":"unavailable","bucket":"UNKNOWN","rows":0,"sessions":0,
            "direction_accuracy_pct":np.nan,"close_mape_pct":np.nan,"status":"NO_NEWS_FEATURES"}])
    use=["prediction_date","symbol"]+signals
    x=e.merge(c[use].drop_duplicates(["prediction_date","symbol"],keep="last"),on=["prediction_date","symbol"],how="inner")
    numeric(x,signals+["close_direction_correct","close_abs_pct_error"])
    rows=[]
    for signal in signals:
        vals=x[signal]
        buckets=np.select([vals.gt(.05),vals.lt(-.05)],["POSITIVE","NEGATIVE"],default="NEUTRAL_OR_MISSING")
        x["_bucket"]=buckets
        for bucket,g in x.groupby("_bucket"):
            d=pd.to_numeric(g.get("close_direction_correct"),errors="coerce").dropna()
            m=pd.to_numeric(g.get("close_abs_pct_error"),errors="coerce").dropna()
            rows.append({"news_signal":signal,"bucket":bucket,"rows":len(g),
                "sessions":int(g["prediction_date"].nunique()),
                "direction_accuracy_pct":float(d.mean()*100) if len(d) else np.nan,
                "close_mape_pct":float(m.mean()*100) if len(m) else np.nan,
                "status":"EVIDENCE_ONLY" if len(g)>=30 else "INSUFFICIENT_SAMPLE"})
    return pd.DataFrame(rows)

def run() -> dict:
    DATA.mkdir(exist_ok=True)
    evaluations=read_csv("evaluations.csv")
    candidates=read_csv("prediction_candidates_history.csv")
    history=read_csv("ohlcv.csv")
    v1=read_csv("portfolio_daily.csv"); v2=read_csv("portfolio_v2_daily.csv")
    t1=read_csv("paper_trades.csv"); t2=read_csv("paper_trades_v2.csv")
    calibration=direction_calibration(evaluations,candidates)
    leak=leakage_audit(evaluations,candidates,history)
    evidence=matched_evidence(v1,v2,t1,t2)
    drift=drift_report(evaluations)
    news=news_impact_report(evaluations,candidates)
    calibration.to_csv(DIRECTION,index=False); leak.to_csv(LEAKAGE,index=False)
    evidence.to_csv(AB,index=False); drift.to_csv(DRIFT,index=False); news.to_csv(NEWS,index=False)
    latest=evidence.iloc[-1].to_dict() if not evidence.empty else {}
    leak_warnings=int(leak["status"].isin(["WARN","BLOCKED"]).sum()) if not leak.empty else 1
    summary={"phases":[21,22,23,24,25,26],"as_of":str(pd.Timestamp.now().date()),
      "phase21":{"calibration_buckets":len(calibration),"evaluation_rows":len(evaluations)},
      "phase22":{"checks":len(leak),"warnings_or_blocks":leak_warnings,"status":"REVIEW" if leak_warnings else "CHECKS_PASS"},
      "phase23":{"matched_sessions":int(latest.get("common_sessions",0) or 0),"v1_trade_count":int(latest.get("v1_trade_count",0) or 0),
                 "v2_trade_count":int(latest.get("v2_trade_count",0) or 0),"evidence_gate":latest.get("evidence_gate","COLLECTING_MATCHED_SESSIONS"),
                 "production_champion":"V1"},
      "phase24":{"status":"INCREMENTAL_HISTORY_MERGE_ALREADY_USED_BY_PIPELINE",
                 "history_rows":len(history),"duplicate_date_symbol_rows":int(history.assign(date=pd.to_datetime(history.get("date"),errors="coerce").dt.normalize(),symbol=history.get("symbol",pd.Series(dtype=str)).astype(str).str.upper().str.strip()).duplicated(["date","symbol"]).sum()) if not history.empty and {"date","symbol"}.issubset(history.columns) else None},
      "phase25":{"drift_status":str(drift.iloc[0].get("drift_status","INSUFFICIENT_DATA")) if not drift.empty else "INSUFFICIENT_DATA"},
      "phase26":{"news_rows":len(news),"status":"ANALYSIS_ONLY"},
      "safety":{"production_model_changed":False,"automatic_rollback":False,"automatic_promotion":False,"v1_remains_champion":True}}
    SUMMARY.write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    return summary

if __name__=="__main__":
    run()
