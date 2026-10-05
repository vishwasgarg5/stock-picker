from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "performance_audit.csv"
SUMMARY = DATA / "performance_audit_summary.csv"

def read(name, date_col=None):
    p=DATA/name
    if not p.exists(): return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=[date_col] if date_col else None)
    except Exception:
        return pd.DataFrame()

def _metric_block(e):
    if e.empty: return {}
    out={"evaluation_rows":len(e)}
    for c in ["close_mape_pct","open_mape_pct","high_mape_pct","low_mape_pct",
              "direction_accuracy_pct","baseline_close_mape_pct"]:
        if c in e:
            v=pd.to_numeric(e[c],errors="coerce").dropna()
            if not v.empty: out[c]=float(v.mean())
    return out

def run_audit():
    e=read("evaluations.csv","target_date")
    v1=read("portfolio_daily.csv","target_date")
    v2=read("portfolio_v2_daily.csv","target_date")
    conf=read("confidence_validation_summary.csv","as_of")
    rank=read("ranking_model_validation_summary.csv")
    ab=read("strategy_ab_comparison.csv","as_of")

    rows=[]
    m=_metric_block(e)
    for k,v in m.items(): rows.append({"area":"OHLC/model","metric":k,"value":v})

    if not v1.empty and "daily_profit_loss" in v1:
        rows += [
            {"area":"V1","metric":"sessions","value":len(v1)},
            {"area":"V1","metric":"net_pnl","value":pd.to_numeric(v1.daily_profit_loss,errors="coerce").sum()},
            {"area":"V1","metric":"max_drawdown_pct","value":pd.to_numeric(v1.get("drawdown_pct",pd.Series(dtype=float)),errors="coerce").min()},
        ]
    if not v2.empty and "daily_profit_loss" in v2:
        rows += [
            {"area":"V2","metric":"sessions","value":len(v2)},
            {"area":"V2","metric":"net_pnl","value":pd.to_numeric(v2.daily_profit_loss,errors="coerce").sum()},
            {"area":"V2","metric":"max_drawdown_pct","value":pd.to_numeric(v2.get("drawdown_pct",pd.Series(dtype=float)),errors="coerce").min()},
            {"area":"V2","metric":"trades","value":pd.to_numeric(v2.get("trades",pd.Series(dtype=float)),errors="coerce").sum()},
        ]
    if not conf.empty:
        r=conf.iloc[-1]
        for c in ["rows","sessions","top_confidence_mape_pct","bottom_confidence_mape_pct",
                  "top_bottom_mape_improvement_pct","top_confidence_direction_pct",
                  "bottom_confidence_direction_pct","confidence_promotion_evidence"]:
            if c in r: rows.append({"area":"Confidence","metric":c,"value":r[c]})
    if not rank.empty:
        r=rank.iloc[-1]
        for c in ["sessions","rows","mean_top10_return_lift_pct","positive_rate_lift_pct","promotion_evidence","status"]:
            if c in r: rows.append({"area":"Ranking challenger","metric":c,"value":r[c]})
    if not ab.empty:
        r=ab.iloc[-1]
        for c in ["common_sessions","v2_trades","pnl_lift","return_lift_pct","session_win_rate_pct","production_strategy","status"]:
            if c in r: rows.append({"area":"Strategy A/B","metric":c,"value":r[c]})

    audit=pd.DataFrame(rows)
    audit["as_of"]=pd.Timestamp.now().normalize()
    audit.to_csv(OUT,index=False)

    findings=[]
    if m.get("close_mape_pct") is not None and m.get("baseline_close_mape_pct") is not None:
        findings.append("MODEL_BEATS_BASELINE" if m["close_mape_pct"] < m["baseline_close_mape_pct"] else "MODEL_NOT_BEATING_BASELINE")
    if not conf.empty and str(conf.iloc[-1].get("confidence_promotion_evidence","False")).lower()=="true":
        findings.append("CONFIDENCE_READY")
    else: findings.append("CONFIDENCE_NOT_PROMOTED")
    if not rank.empty and str(rank.iloc[-1].get("status",""))=="promote":
        findings.append("RANKING_READY")
    else: findings.append("RANKING_NOT_PROMOTED")
    if not ab.empty:
        findings.append("V2_CURRENTLY_"+str(ab.iloc[-1].get("status","UNKNOWN")).upper())
    summary=pd.DataFrame([{"as_of":pd.Timestamp.now().normalize(),"findings":" | ".join(findings),
                           "audit_rows":len(audit)}])
    summary.to_csv(SUMMARY,index=False)
    print(summary.to_string(index=False))
    return audit

if __name__=="__main__":
    run_audit()
