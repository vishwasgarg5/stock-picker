from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE_FILE = DATA / "strategy_state.json"
HISTORY_FILE = DATA / "strategy_promotion_history.csv"

V1_FILE = DATA / "portfolio_daily.csv"
V2_FILE = DATA / "portfolio_v2_daily.csv"

MIN_SESSIONS = 12
MIN_V2_TRADES = 30
MIN_IMPROVEMENT = 0.05
MAX_V2_DRAWDOWN_DEGRADATION = 1.0


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["target_date"])
    except Exception:
        return pd.DataFrame()


def _metrics(path: Path) -> dict:
    x = _load(path)
    if x.empty:
        return {"sessions": 0, "trades": 0, "net_pnl": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0}

    trades = int(x["trades"].sum()) if "trades" in x else 0
    pnl = float(x["daily_profit_loss"].sum()) if "daily_profit_loss" in x else 0.0
    ret = float(x.iloc[-1].get("cumulative_return_pct", pnl / 1000.0))
    dd = float(x["drawdown_pct"].min()) if "drawdown_pct" in x else 0.0

    return {
        "sessions": int(len(x)),
        "trades": trades,
        "net_pnl": pnl,
        "return_pct": ret,
        "max_drawdown_pct": dd,
        "win_rate_pct": float("nan"),
    }


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"production_strategy": "V1", "status": "collecting", "reason": "initial_state"}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"production_strategy": "V1", "status": "collecting", "reason": "invalid_state"}


def evaluate_promotion() -> dict:
    v1 = _metrics(V1_FILE)
    v2 = _metrics(V2_FILE)
    state = _load_state()

    v2_sessions = v2["sessions"]
    enough = v2_sessions >= MIN_SESSIONS and v2["trades"] >= MIN_V2_TRADES

    pnl_lift = v2["net_pnl"] - v1["net_pnl"]
    return_lift = v2["return_pct"] - v1["return_pct"]
    # Drawdown is negative; e.g. -2% is better than -4%.
    dd_better = v2["max_drawdown_pct"] >= v1["max_drawdown_pct"] - MAX_V2_DRAWDOWN_DEGRADATION

    promote = bool(
        enough
        and pnl_lift > 0
        and return_lift >= MIN_IMPROVEMENT
        and dd_better
    )

    current = str(state.get("production_strategy", "V1"))
    # Promotion is deliberately one-way only after robust evidence. If V2 is
    # promoted later, subsequent degradation is handled by the rollback gate.
    if current == "V1" and promote:
        production = "V2"
        status = "promoted"
        reason = "V2 beats V1 on net P/L, return and drawdown"
    elif current == "V2":
        # Roll back only on persistent deterioration, not one bad session.
        severe = enough and (
            v2["net_pnl"] < v1["net_pnl"]
            and v2["return_pct"] < v1["return_pct"] - MIN_IMPROVEMENT
            and v2["max_drawdown_pct"] < v1["max_drawdown_pct"] - MAX_V2_DRAWDOWN_DEGRADATION
        )
        production = "V1" if severe else "V2"
        status = "rolled_back" if severe else "stable"
        reason = "persistent V2 underperformance" if severe else "V2 remains within risk gate"
    else:
        production = "V1"
        status = "collecting" if not enough else "hold"
        reason = "insufficient V2 evidence" if not enough else "V2 has not beaten promotion gates"

    result = {
        "production_strategy": production,
        "status": status,
        "reason": reason,
        "v1": v1,
        "v2": v2,
        "pnl_lift": pnl_lift,
        "return_lift_pct": return_lift,
        "drawdown_gate": dd_better,
        "minimum_sessions": MIN_SESSIONS,
        "minimum_v2_trades": MIN_V2_TRADES,
    }

    STATE_FILE.write_text(json.dumps(result, indent=2, default=str))
    row = {
        "as_of": pd.Timestamp.now().normalize(),
        "production_strategy": production,
        "status": status,
        "reason": reason,
        "v1_net_pnl": v1["net_pnl"],
        "v2_net_pnl": v2["net_pnl"],
        "v1_return_pct": v1["return_pct"],
        "v2_return_pct": v2["return_pct"],
        "v1_max_drawdown_pct": v1["max_drawdown_pct"],
        "v2_max_drawdown_pct": v2["max_drawdown_pct"],
        "v2_sessions": v2_sessions,
        "v2_trades": v2["trades"],
        "return_lift_pct": return_lift,
        "pnl_lift": pnl_lift,
    }
    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HISTORY_FILE, index=False)
    print(f"Strategy gate: production={production}, status={status}, reason={reason}")
    return result


if __name__ == "__main__":
    evaluate_promotion()
