from src.pipeline import apply_risk_off_selection
import pandas as pd

def test_risk_off_prefers_defensive_strength():
    x=pd.DataFrame({"symbol":["A","B","C"],"rank":[1,2,3],"total_score":[50,49.5,49],
                    "return_20d":[1,2,3],"close_sma20_gap":[.01,.02,.03],
                    "volatility20":[.08,.03,.02],"fundamental_score":[10,12,14]})
    out,method=apply_risk_off_selection(x,"RISK_OFF",2)
    assert method=="risk_off_defensive_v1"
    assert out.head(2)["symbol"].tolist()==["C","B"]

def test_non_risk_off_preserves_order():
    x=pd.DataFrame({"symbol":["A","B"],"rank":[1,2],"total_score":[10,9]})
    out,method=apply_risk_off_selection(x,"NEUTRAL",2)
    assert method=="ranking_top10"
    assert out["symbol"].tolist()==["A","B"]
