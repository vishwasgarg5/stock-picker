from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AB_FILE = DATA / "strategy_ab_comparison.csv"


def build_report() -> pd.DataFrame:
    if not AB_FILE.exists():
        return pd.DataFrame()
    x = pd.read_csv(AB_FILE)
    if x.empty:
        return x
    cols = [
        "as_of", "production_strategy", "status", "common_sessions", "v2_trades",
        "v1_net_pnl", "v2_net_pnl", "pnl_lift", "v1_return_pct", "v2_return_pct",
        "return_lift_pct", "v1_max_drawdown_pct", "v2_max_drawdown_pct",
        "session_win_rate_pct", "drawdown_gate",
    ]
    out = x[[c for c in cols if c in x.columns]].copy()
    out.to_csv(DATA / "strategy_ab_report.csv", index=False)
    return out


if __name__ == "__main__":
    print(build_report().to_string(index=False) if not build_report().empty else "No A/B data yet.")
