from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE_FILE = DATA / "strategy_state.json"
HISTORY_FILE = DATA / "strategy_promotion_history.csv"
AB_FILE = DATA / "strategy_ab_comparison.csv"

V1_FILE = DATA / "portfolio_daily.csv"
V2_FILE = DATA / "portfolio_v2_daily.csv"

MIN_COMMON_SESSIONS = 20
MIN_V2_TRADES = 50
MIN_RETURN_LIFT_PCT = 0.05
MAX_DRAWDOWN_DEGRADATION_PCT = 1.0
MIN_SESSION_WIN_RATE_PCT = 60.0


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["target_date"])
    except Exception:
        return pd.DataFrame()


def _matched(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    if v1.empty or v2.empty:
        return pd.DataFrame()
    a = v1[["target_date", "daily_profit_loss", "daily_return_pct", "drawdown_pct"]].copy()
    b = v2[["target_date", "daily_profit_loss", "daily_return_pct", "drawdown_pct", "trades"]].copy()
    a = a.rename(columns={"daily_profit_loss": "v1_pnl", "daily_return_pct": "v1_return_pct", "drawdown_pct": "v1_drawdown_pct"})
    b = b.rename(columns={"daily_profit_loss": "v2_pnl", "daily_return_pct": "v2_return_pct", "drawdown_pct": "v2_drawdown_pct", "trades": "v2_trades"})
    x = a.merge(b, on="target_date", how="inner").sort_values("target_date")
    if x.empty:
        return x
    x["pnl_difference"] = x["v2_pnl"] - x["v1_pnl"]
    x["return_difference_pct"] = x["v2_return_pct"] - x["v1_return_pct"]
    x["v2_wins_session"] = (x["pnl_difference"] > 0).astype(int)
    return x


def _metrics(x: pd.DataFrame, prefix: str) -> dict:
    if x.empty:
        return {"sessions": 0, "trades": 0, "net_pnl": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0}
    pnl = float(x[f"{prefix}_pnl"].sum())
    ret = float(x[f"{prefix}_return_pct"].sum())
    dd = float(x[f"{prefix}_drawdown_pct"].min())
    trades = int(x["v2_trades"].sum()) if prefix == "v2" and "v2_trades" in x else 0
    return {"sessions": int(len(x)), "trades": trades, "net_pnl": pnl, "return_pct": ret, "max_drawdown_pct": dd}


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"production_strategy": "V1", "status": "collecting", "reason": "initial_state"}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"production_strategy": "V1", "status": "collecting", "reason": "invalid_state"}


def evaluate_promotion() -> dict:
    v1_all = _load(V1_FILE)
    v2_all = _load(V2_FILE)
    matched = _matched(v1_all, v2_all)
    state = _load_state()

    common_sessions = len(matched)
    v2_trades = int(matched["v2_trades"].sum()) if not matched.empty else 0
    enough = common_sessions >= MIN_COMMON_SESSIONS and v2_trades >= MIN_V2_TRADES

    if enough:
        v1 = _metrics(matched, "v1")
        v2 = _metrics(matched, "v2")
        return_lift = v2["return_pct"] - v1["return_pct"]
        pnl_lift = v2["net_pnl"] - v1["net_pnl"]
        session_win_rate = float(matched["v2_wins_session"].mean() * 100.0)
        dd_gate = v2["max_drawdown_pct"] >= v1["max_drawdown_pct"] - MAX_DRAWDOWN_DEGRADATION_PCT
    else:
        v1 = _metrics(matched, "v1") if not matched.empty else {}
        v2 = _metrics(matched, "v2") if not matched.empty else {}
        return_lift = pnl_lift = 0.0
        session_win_rate = 0.0
        dd_gate = False

    promote = bool(
        enough
        and pnl_lift > 0
        and return_lift >= MIN_RETURN_LIFT_PCT
        and session_win_rate >= MIN_SESSION_WIN_RATE_PCT
        and dd_gate
    )

    current = str(state.get("production_strategy", "V1"))
    if current == "V1" and promote:
        production, status, reason = "V2", "promoted", "V2 passed matched-date A/B gates"
    elif current == "V2":
        rollback = bool(
            enough
            and pnl_lift < 0
            and return_lift < -MIN_RETURN_LIFT_PCT
            and session_win_rate < 40.0
            and not dd_gate
        )
        production = "V1" if rollback else "V2"
        status = "rolled_back" if rollback else "stable"
        reason = "matched-date V2 deterioration triggered rollback" if rollback else "V2 remains within rollback gate"
    else:
        production = "V1"
        status = "collecting" if not enough else "hold"
        reason = "insufficient matched-date evidence" if not enough else "V2 has not passed all promotion gates"

    comparison = {
        "as_of": pd.Timestamp.now().normalize(),
        "production_strategy": production,
        "status": status,
        "reason": reason,
        "common_sessions": common_sessions,
        "v2_trades": v2_trades,
        "v1_net_pnl": v1.get("net_pnl", 0.0),
        "v2_net_pnl": v2.get("net_pnl", 0.0),
        "pnl_lift": pnl_lift,
        "v1_return_pct": v1.get("return_pct", 0.0),
        "v2_return_pct": v2.get("return_pct", 0.0),
        "return_lift_pct": return_lift,
        "v1_max_drawdown_pct": v1.get("max_drawdown_pct", 0.0),
        "v2_max_drawdown_pct": v2.get("max_drawdown_pct", 0.0),
        "session_win_rate_pct": session_win_rate,
        "drawdown_gate": dd_gate,
    }

    pd.DataFrame([comparison]).to_csv(AB_FILE, index=False)
    STATE_FILE.write_text(json.dumps(comparison, indent=2, default=str))

    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([comparison])], ignore_index=True).to_csv(HISTORY_FILE, index=False)

    if not matched.empty:
        matched.to_csv(DATA / "strategy_ab_sessions.csv", index=False)

    print(
        f"Strategy A/B: common_sessions={common_sessions}, v2_trades={v2_trades}, "
        f"session_win_rate={session_win_rate:.1f}%, production={production}, status={status}"
    )
    return comparison


if __name__ == "__main__":
    evaluate_promotion()
