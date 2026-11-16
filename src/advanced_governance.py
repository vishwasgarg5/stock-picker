from __future__ import annotations

"""Advanced selection, calibration and promotion controls (steps 9-19).

All controls are deterministic and point-in-time safe. They enrich the existing
V1 pipeline; promotion remains evidence-gated and V1-safe by default.
"""

from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
CONFIG=DATA/"advanced_selection_config.json"
SUMMARY=DATA/"advanced_selection_summary.csv"

def confidence_tier(score: float) -> str:
    s=float(score) if np.isfinite(float(score)) else 0.0
    return "HIGH" if s>=75 else "MEDIUM" if s>=55 else "LOW"

def calibrate_confidence(confidence: pd.Series, correct: pd.Series) -> dict:
    p=pd.to_numeric(confidence,errors="coerce").clip(0,100)/100.0
    y=pd.to_numeric(correct,errors="coerce").clip(0,1)
    x=pd.DataFrame({"p":p,"y":y}).dropna()
    if x.empty:
        return {"rows":0,"brier_score":np.nan,"ece":np.nan,"calibration_status":"collecting"}
    brier=float(((x.p-x.y)**2).mean())
    bins=pd.cut(x.p,bins=[0,.2,.4,.6,.8,1.0],include_lowest=True)
    ece=0.0
    for _,g in x.groupby(bins,observed=False):
        if len(g):
            ece += len(g)/len(x)*abs(float(g.p.mean())-float(g.y.mean()))
    return {"rows":int(len(x)),"brier_score":brier,"ece":float(ece),
            "calibration_status":"PASS" if brier<=0.25 and ece<=0.15 else "WARNING"}

def stock_news_impact(headlines: list[dict], universe: pd.DataFrame) -> pd.DataFrame:
    """Match company-specific headlines using symbols and distinctive company tokens."""
    if universe.empty or "symbol" not in universe.columns:
        return pd.DataFrame(columns=["symbol","stock_news_score","stock_news_count","stock_news_reason"])
    from .news_context import classify_headline
    u=universe.copy()
    u["symbol"]=u["symbol"].astype(str).str.upper().str.strip()
    u["company_name"]=u.get("company_name",u["symbol"]).astype(str)
    token_freq={}
    token_sets={}
    stop={"india","limited","ltd","company","holdings","group","industries","services","financial","solutions","enterprise","enterprises"}
    for _,r in u.iterrows():
        tokens={w.lower() for w in re.findall(r"[a-zA-Z0-9]+",str(r["company_name"])) if len(w)>=8 and w.lower() not in stop}
        token_sets[str(r["symbol"])]=tokens
        for t in tokens: token_freq[t]=token_freq.get(t,0)+1
    rows=[]
    for _,r in u.iterrows():
        sym=str(r["symbol"])
        name=str(r["company_name"]).lower()
        distinctive={t for t in token_sets.get(sym,set()) if token_freq.get(t,0)==1}
        matches=[]
        for item in headlines:
            h=str(item.get("headline","")).lower()
            if sym.lower() in h or name in h or any(t in h for t in distinctive):
                c=classify_headline(h)
                matches.append((float(c["market_impact"]),str(item.get("headline","")),c["event"]))
        if matches:
            score=float(np.clip(np.mean([m[0] for m in matches]),-1,1))
            reason=max(matches,key=lambda z:abs(z[0]))[1]
        else:
            score=0.0; reason=""
        rows.append({"symbol":sym,"stock_news_score":score,"stock_news_count":len(matches),"stock_news_reason":reason})
    return pd.DataFrame(rows)

def regime_weights(regime: str, risk_level: str="NEUTRAL") -> dict:
    r=str(regime).upper()
    if str(risk_level).upper()=="RISK_OFF" or r in {"BEAR","RISK_OFF"}:
        return {"technical":0.55,"fundamental":0.25,"defensive":0.15,"news":0.05}
    if r in {"BULL","RISK_ON"}:
        return {"technical":0.60,"fundamental":0.20,"defensive":0.05,"news":0.15}
    return {"technical":0.50,"fundamental":0.30,"defensive":0.10,"news":0.10}

def adaptive_selection_score(df: pd.DataFrame, regime: str, risk_level: str) -> pd.Series:
    x=df.copy()
    for c in ["technical_score","fundamental_score","risk_off_defensive_score","sector_news_score","stock_news_score"]:
        if c not in x.columns: x[c]=0.0
        x[c]=pd.to_numeric(x[c],errors="coerce").fillna(0.0)
    w=regime_weights(regime,risk_level)
    def norm(s):
        lo,hi=float(s.min()),float(s.max())
        return pd.Series(50.0,index=s.index) if hi-lo<1e-12 else (s-lo)/(hi-lo)*100.0
    return (w["technical"]*norm(x.technical_score)
            +w["fundamental"]*norm(x.fundamental_score)
            +w["defensive"]*norm(x.risk_off_defensive_score)
            +w["news"]*norm(x.sector_news_score+x.stock_news_score*2.0))

def risk_budget(df: pd.DataFrame, total_capital: float=100000.0, max_position_pct: float=0.20) -> pd.DataFrame:
    x=df.copy()
    vol=pd.to_numeric(x.get("volatility20",pd.Series(0.02,index=x.index)),errors="coerce").fillna(0.02).clip(lower=0.005)
    conf=pd.to_numeric(x.get("adjusted_confidence_score",pd.Series(50.0,index=x.index)),errors="coerce").fillna(50)/100
    raw=(conf/(vol*100)).replace([np.inf,-np.inf],np.nan).fillna(0)
    weights=(raw/raw.sum()).fillna(0.0)
    for _ in range(10):
        over=weights>max_position_pct
        if not over.any(): break
        excess=float((weights[over]-max_position_pct).sum())
        weights[over]=max_position_pct
        under=~over
        room=(max_position_pct-weights[under]).clip(lower=0)
        if excess<=1e-12 or not under.any() or room.sum()<=1e-12: break
        weights.loc[under]=weights.loc[under]+excess*(room/room.sum())
    x["risk_budget_weight"]=weights
    x["risk_budget_value"]=weights*float(total_capital)
    return x

def drawdown_guard(pnl: pd.Series, max_drawdown_pct: float=8.0) -> dict:
    x=pd.to_numeric(pnl,errors="coerce").dropna()
    if x.empty: return {"max_drawdown_pct":0.0,"trade_allowed":True,"reason":"collecting"}
    equity=x.cumsum()
    peak=equity.cummax()
    dd=(equity-peak)
    base=max(abs(float(equity.iloc[0])) if equity.iloc[0]!=0 else 1.0,1.0)
    dd_pct=float(dd.min()/base*100)
    return {"max_drawdown_pct":dd_pct,"trade_allowed":bool(dd_pct>=-abs(max_drawdown_pct)),
            "reason":"PASS" if dd_pct>=-abs(max_drawdown_pct) else "DRAWDOWN_GUARD"}

def turnover_guard(previous: set[str], current: set[str], max_turnover: float=0.60) -> dict:
    p=set(previous); c=set(current)
    denom=max(len(p|c),1)
    turnover=len(p.symmetric_difference(c))/denom
    return {"turnover":float(turnover),"trade_allowed":turnover<=max_turnover,
            "reason":"PASS" if turnover<=max_turnover else "TURNOVER_GUARD"}

def data_freshness(hist: pd.DataFrame, expected_days: int=2) -> dict:
    if hist.empty or "date" not in hist.columns:
        return {"status":"FAIL","age_days":9999}
    d=pd.to_datetime(hist["date"],errors="coerce").dropna().max()
    if pd.isna(d): return {"status":"FAIL","age_days":9999}
    age=max((pd.Timestamp.now().normalize()-pd.Timestamp(d).normalize()).days,0)
    return {"status":"PASS" if age<=expected_days else "WARNING","age_days":int(age),"latest_date":str(pd.Timestamp(d).date())}

def explain_selection(row: pd.Series) -> dict:
    return {
        "symbol":str(row.get("symbol","")),
        "rank":int(pd.to_numeric(row.get("rank",0),errors="coerce") or 0),
        "regime":str(row.get("market_regime",row.get("index_market_regime","NEUTRAL"))),
        "risk_level":str(row.get("risk_level","NEUTRAL")),
        "confidence":float(pd.to_numeric(row.get("adjusted_confidence_score",0),errors="coerce") or 0),
        "stock_news_score":float(pd.to_numeric(row.get("stock_news_score",0),errors="coerce") or 0),
        "sector_news_score":float(pd.to_numeric(row.get("sector_news_score",0),errors="coerce") or 0),
        "selection_reason":"highest adaptive score after regime, quality, risk and contextual-news controls",
    }

def promotion_gate(metrics: dict) -> dict:
    checks={
        "minimum_sessions":int(metrics.get("sessions",0))>=20,
        "minimum_trades":int(metrics.get("trades",0))>=50,
        "return_lift":float(metrics.get("return_lift_pct",-999))>=0.05,
        "session_win_rate":float(metrics.get("session_win_rate_pct",0))>=60,
        "ci_low":float(metrics.get("return_lift_ci_low_pct",-999))>=0,
        "confidence_evidence":bool(metrics.get("confidence_evidence",False)),
        "governance_safe":bool(metrics.get("governance_safe",False)),
        "no_drawdown_breach":bool(metrics.get("no_drawdown_breach",False)),
    }
    passed=all(checks.values())
    return {"promotion_eligible":passed,"checks":checks,
            "decision":"PROMOTE_CANDIDATE" if passed else "KEEP_V1"}

def rollback_decision(state: dict, candidate_metrics: dict, rollback_lift_pct: float=-0.25) -> dict:
    lift=float(candidate_metrics.get("return_lift_pct",0.0))
    if str(state.get("production_strategy","V1")).upper()=="V2" and lift<rollback_lift_pct:
        return {"rollback":True,"target_strategy":"V1","reason":"candidate_return_lift_below_rollback_threshold"}
    return {"rollback":False,"target_strategy":state.get("production_strategy","V1"),"reason":"no_rollback_triggered"}

def build_summary(hist: pd.DataFrame, candidates: pd.DataFrame | None=None) -> dict:
    freshness=data_freshness(hist)
    confidence={}
    if candidates is not None and not candidates.empty and "adjusted_confidence_score" in candidates.columns:
        confidence={"high_confidence_count":int((pd.to_numeric(candidates["adjusted_confidence_score"],errors="coerce")>=75).sum()),
                     "medium_confidence_count":int(((pd.to_numeric(candidates["adjusted_confidence_score"],errors="coerce")>=55)&(pd.to_numeric(candidates["adjusted_confidence_score"],errors="coerce")<75)).sum()),
                     "low_confidence_count":int((pd.to_numeric(candidates["adjusted_confidence_score"],errors="coerce")<55).sum())}
    out={"as_of":pd.Timestamp.now(tz="UTC").isoformat(),"advanced_control_version":"advanced_v1",
         "data_freshness":freshness,**confidence}
    CONFIG.write_text(json.dumps(out,indent=2,default=str))
    pd.DataFrame([out]).to_csv(SUMMARY,index=False)
    return out

def run_advanced_controls() -> dict:
    hist=pd.read_csv(DATA/"ohlcv.csv") if (DATA/"ohlcv.csv").exists() else pd.DataFrame()
    candidates=pd.read_csv(DATA/"prediction_candidates.csv") if (DATA/"prediction_candidates.csv").exists() else pd.DataFrame()
    summary=build_summary(hist,candidates)
    gov=pd.read_csv(DATA/"model_governance_summary.csv") if (DATA/"model_governance_summary.csv").exists() else pd.DataFrame()
    ab=pd.read_csv(DATA/"strategy_ab_comparison.csv") if (DATA/"strategy_ab_comparison.csv").exists() else pd.DataFrame()
    conf=pd.read_csv(DATA/"confidence_validation_summary.csv") if (DATA/"confidence_validation_summary.csv").exists() else pd.DataFrame()
    if not ab.empty:
        row=ab.iloc[-1]
        metrics={
            "sessions":int(pd.to_numeric(row.get("common_sessions",0),errors="coerce") or 0),
            "trades":int(pd.to_numeric(row.get("v2_trades",0),errors="coerce") or 0),
            "return_lift_pct":float(pd.to_numeric(row.get("return_lift_pct",-999),errors="coerce")),
            "session_win_rate_pct":float(pd.to_numeric(row.get("session_win_rate_pct",0),errors="coerce")),
            "return_lift_ci_low_pct":float(pd.to_numeric(row.get("return_lift_ci_low_pct",-999),errors="coerce")),
            "confidence_evidence":bool(not conf.empty and str(conf.iloc[-1].get("confidence_promotion_evidence",False)).lower()=="true"),
            "governance_safe":bool(not gov.empty and str(gov.iloc[-1].get("safety_status","FAIL")).upper()=="PASS"),
            "no_drawdown_breach":bool(float(pd.to_numeric(row.get("v2_max_drawdown_pct",0),errors="coerce") or 0)>=-8.0),
        }
    else:
        metrics={"sessions":0,"trades":0,"return_lift_pct":-999,"session_win_rate_pct":0,
                 "return_lift_ci_low_pct":-999,"confidence_evidence":False,
                 "governance_safe":False,"no_drawdown_breach":False}
    gate=promotion_gate(metrics)
    out={**summary,"promotion_gate":gate,"production_strategy":"V1"}
    pd.DataFrame([{"as_of":out["as_of"],"promotion_eligible":gate["promotion_eligible"],
                   "decision":gate["decision"],"checks":json.dumps(gate["checks"],sort_keys=True)}]
                ).to_csv(DATA/"advanced_promotion_gate.csv",index=False)
    CONFIG.write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps(out,indent=2,default=str))
    return out

if __name__=="__main__":
    run_advanced_controls()
