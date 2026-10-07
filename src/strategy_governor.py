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
CONFIDENCE_SUMMARY_FILE = DATA / "confidence_validation_summary.csv"
PHASE2_SUMMARY_FILE = DATA / "phase2_performance_summary.csv"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["target_date"])
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _matched(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    if v1.empty or v2.empty:
        return pd.DataFrame()
    needed1 = ["target_date", "daily_profit_loss", "daily_return_pct", "drawdown_pct"]
    needed2 = ["target_date", "daily_profit_loss", "daily_return_pct", "drawdown_pct", "trades"]
    if not set(needed1).issubset(v1.columns) or not set(needed2).issubset(v2.columns):
        return pd.DataFrame()
    a = v1[needed1].copy()
    b = v2[needed2].copy()
    a["target_date"] = pd.to_datetime(a["target_date"], errors="coerce").dt.normalize()
    b["target_date"] = pd.to_datetime(b["target_date"], errors="coerce").dt.normalize()
    a = a.rename(columns={"daily_profit_loss": "v1_pnl", "daily_return_pct": "v1_return_pct", "drawdown_pct": "v1_drawdown_pct"})
    b = b.rename(columns={"daily_profit_loss": "v2_pnl", "daily_return_pct": "v2_return_pct", "drawdown_pct": "v2_drawdown_pct", "trades": "v2_trades"})
    x = a.merge(b, on="target_date", how="inner").dropna(subset=["target_date"]).sort_values("target_date")
    if x.empty:
        return x
    x["pnl_difference"] = x["v2_pnl"] - x["v1_pnl"]
    x["return_difference_pct"] = x["v2_return_pct"] - x["v1_return_pct"]
    x["v2_wins_session"] = (x["pnl_difference"] > 0).astype(int)
    return x


def _metrics(x: pd.DataFrame, prefix: str) -> dict:
    if x.empty:
        return {"sessions": 0, "trades": 0, "net_pnl": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0}
    pnl = float(pd.to_numeric(x[f"{prefix}_pnl"], errors="coerce").fillna(0).sum())
    ret = float(pd.to_numeric(x[f"{prefix}_return_pct"], errors="coerce").fillna(0).sum())
    dd = float(pd.to_numeric(x[f"{prefix}_drawdown_pct"], errors="coerce").min())
    trades = int(pd.to_numeric(x["v2_trades"], errors="coerce").fillna(0).sum()) if prefix == "v2" and "v2_trades" in x else 0
    return {"sessions": int(len(x)), "trades": trades, "net_pnl": pnl, "return_pct": ret, "max_drawdown_pct": dd}


def _confidence_promoted() -> bool:
    if not CONFIDENCE_SUMMARY_FILE.exists():
        return False
    try:
        x = pd.read_csv(CONFIDENCE_SUMMARY_FILE)
        if x.empty:
            return False
        return str(x.iloc[-1].get("confidence_promotion_evidence", False)).strip().lower() == "true"
    except Exception:
        return False


def _phase2_enabled() -> tuple[bool, dict]:
    if not PHASE2_SUMMARY_FILE.exists():
        return False, {}
    try:
        x = pd.read_csv(PHASE2_SUMMARY_FILE)
        if x.empty:
            return False, {}
        row = x.iloc[-1].to_dict()
        enabled = str(row.get("production_enabled", False)).strip().lower() == "true"
        return enabled, row
    except Exception:
        return False, {}


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
    confidence_promoted = _confidence_promoted()
    phase2_enabled, phase2 = _phase2_enabled()
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
        and phase2_enabled
        and confidence_promoted
        and pnl_lift > 0
        and return_lift >= MIN_RETURN_LIFT_PCT
        and session_win_rate >= MIN_SESSION_WIN_RATE_PCT
        and dd_gate
    )

    current = str(state.get("production_strategy", "V1"))
    if current == "V1" and promote:
        production, status, reason = "V2", "promoted", "V2 passed matched-date, confidence and Phase-2 gates"
    elif current == "V2":
        rollback = bool(
            not phase2_enabled
            or not confidence_promoted
            or (
                enough
                and pnl_lift < 0
                and return_lift < -MIN_RETURN_LIFT_PCT
                and session_win_rate < 40.0
                and not dd_gate
            )
        )
        production = "V1" if rollback else "V2"
        status = "rolled_back" if rollback else "stable"
        reason = "Phase-2/confidence safety gate failed or matched-date deterioration triggered rollback" if rollback else "V2 remains within rollback gates"
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
        "confidence_promoted": confidence_promoted,
        "phase2_enabled": phase2_enabled,
        "phase2_direction_lift_pct": phase2.get("direction_lift_pct", 0.0),
        "phase2_shadow_return_lift_pct": phase2.get("return_lift_pct", 0.0),
    }

    pd.DataFrame([comparison]).to_csv(AB_FILE, index=False)
    STATE_FILE.write_text(json.dumps(comparison, indent=2, default=str))

    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([comparison])], ignore_index=True).to_csv(HISTORY_FILE, index=False)

    if not matched.empty:
        matched.to_csv(DATA / "strategy_ab_sessions.csv", index=False)

    print(
        f"Strategy A/B: common_sessions={common_sessions}, v2_trades={v2_trades}, "
        f"phase2={phase2_enabled}, session_win_rate={session_win_rate:.1f}%, "
        f"production={production}, status={status}"
    )
    return comparison


if __name__ == "__main__":
    evaluate_promotion()
