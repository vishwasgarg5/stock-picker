"""Build a detailed V1/V2 portfolio comparison report from stored data."""
from pathlib import Path
import json
import pandas as pd

DATA = Path("data")
OUT = DATA / "portfolio_comparison.csv"
SUMMARY = DATA / "portfolio_comparison_summary.csv"

def _read(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def main():
    v1 = _read("portfolio_daily.csv")
    v2 = _read("portfolio_v2_daily.csv")
    trades = _read("paper_trades.csv")
    v2trades = _read("paper_trades_v2.csv")

    rows = []
    if not v1.empty:
        latest = v1.iloc[-1]
        peak = pd.to_numeric(v1["portfolio_value"], errors="coerce").cummax()
        dd = (pd.to_numeric(v1["portfolio_value"], errors="coerce") / peak - 1) * 100
        rows += [
            {"strategy":"V1","metric":"latest_date","value":latest["target_date"]},
            {"strategy":"V1","metric":"portfolio_value","value":latest["portfolio_value"]},
            {"strategy":"V1","metric":"cumulative_return_pct","value":latest["cumulative_return_pct"]},
            {"strategy":"V1","metric":"max_drawdown_pct","value":dd.min()},
            {"strategy":"V1","metric":"sessions","value":len(v1)},
        ]
    v1b = trades[trades["signal"].eq("BUY")].copy() if not trades.empty else pd.DataFrame()
    if not v1b.empty:
        pnl = pd.to_numeric(v1b["profit_loss"], errors="coerce").fillna(0)
        rows += [
            {"strategy":"V1","metric":"buy_trades","value":len(v1b)},
            {"strategy":"V1","metric":"winning_trades","value":int((pnl>0).sum())},
            {"strategy":"V1","metric":"losing_trades","value":int((pnl<0).sum())},
            {"strategy":"V1","metric":"win_rate_pct","value":(pnl>0).mean()*100},
            {"strategy":"V1","metric":"net_pnl","value":pnl.sum()},
        ]
    if not v2.empty:
        latest = v2.iloc[-1]
        rows += [{"strategy":"V2","metric":"latest_date","value":latest["target_date"]},
                 {"strategy":"V2","metric":"portfolio_value","value":latest["portfolio_value"]},
                 {"strategy":"V2","metric":"cumulative_return_pct","value":latest["cumulative_return_pct"]},
                 {"strategy":"V2","metric":"sessions","value":len(v2)}]
    else:
        rows += [{"strategy":"V2","metric":"status","value":"NO_PORTFOLIO_HISTORY"}]
    v2b = v2trades[v2trades["signal"].eq("BUY")].copy() if not v2trades.empty else pd.DataFrame()
    rows += [{"strategy":"V2","metric":"buy_trades","value":len(v2b)},
             {"strategy":"V2","metric":"status","value":"COLLECTING" if len(v2b)<50 else "COMPARABLE"}]
    pd.DataFrame(rows).to_csv(OUT, index=False)

    latest_day = None
    if not v1b.empty and "target_date" in v1b:
        latest_day = v1b["target_date"].max()
        d = v1b[v1b["target_date"].eq(latest_day)]
        day_pnl = pd.to_numeric(d["profit_loss"], errors="coerce").fillna(0)
        detail = {
            "latest_target_date": latest_day,
            "trades": int(len(d)),
            "winners": int((day_pnl>0).sum()),
            "losers": int((day_pnl<0).sum()),
            "net_pnl": float(day_pnl.sum()),
            "symbols": ",".join(d["symbol"].astype(str)),
        }
    else:
        detail = {"latest_target_date":"","trades":0,"winners":0,"losers":0,"net_pnl":0.0,"symbols":""}

    summary = {
        "as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "production_strategy":"V1",
        "latest_v1_date": str(v1["target_date"].iloc[-1]) if not v1.empty else "",
        "v1_portfolio_value": float(v1["portfolio_value"].iloc[-1]) if not v1.empty else 100000.0,
        "v1_cumulative_return_pct": float(v1["cumulative_return_pct"].iloc[-1]) if not v1.empty else 0.0,
        "v1_max_drawdown_pct": float(((pd.to_numeric(v1["portfolio_value"], errors="coerce") / pd.to_numeric(v1["portfolio_value"], errors="coerce").cummax()-1)*100).min()) if not v1.empty else 0.0,
        "v2_buy_trades": int(len(v2b)),
        "comparison_status":"INSUFFICIENT_EVIDENCE" if len(v2b)<50 else "COMPARABLE",
        **detail,
    }
    pd.DataFrame([summary]).to_csv(SUMMARY,index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
