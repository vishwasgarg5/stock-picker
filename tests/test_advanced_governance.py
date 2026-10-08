import pandas as pd
from src.advanced_governance import (
    confidence_tier, calibrate_confidence, regime_weights,
    risk_budget, drawdown_guard, turnover_guard, promotion_gate,
)

def test_confidence_tiers():
    assert confidence_tier(80)=="HIGH"
    assert confidence_tier(60)=="MEDIUM"
    assert confidence_tier(40)=="LOW"

def test_calibration_metrics():
    r=calibrate_confidence(pd.Series([90,80,20,10]),pd.Series([1,1,0,0]))
    assert r["rows"]==4
    assert r["brier_score"]<0.1
    assert r["ece"]<0.1

def test_regime_weights_are_defensive_in_risk_off():
    w=regime_weights("NEUTRAL","RISK_OFF")
    assert w["fundamental"]>=0.20
    assert w["news"]<=0.10

def test_risk_budget_caps_positions():
    x=pd.DataFrame({"volatility20":[.01,.02,.03],"adjusted_confidence_score":[90,70,60]})
    out=risk_budget(x,100000,.20)
    assert out["risk_budget_weight"].max()<=.20+1e-9
    assert abs(out["risk_budget_weight"].sum()-1.0)<1e-9

def test_drawdown_guard():
    r=drawdown_guard(pd.Series([100,90,95]))
    assert r["trade_allowed"] is True

def test_turnover_guard():
    assert turnover_guard({"A","B"},{"A","B"},.60)["trade_allowed"] is True
    assert turnover_guard({"A","B"},{"C","D"},.60)["trade_allowed"] is False

def test_promotion_gate_keeps_v1_without_full_evidence():
    r=promotion_gate({"sessions":20,"trades":50,"return_lift_pct":1,"session_win_rate_pct":70,
                      "return_lift_ci_low_pct":0.2,"confidence_evidence":True,
                      "governance_safe":True,"no_drawdown_breach":True})
    assert r["promotion_eligible"] is True
    r2=promotion_gate({"sessions":20})
    assert r2["promotion_eligible"] is False
