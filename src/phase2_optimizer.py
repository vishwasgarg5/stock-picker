from __future__ import annotations

"""Phase 2 performance optimizer.

All eight Phase-2 improvements run in shadow mode first.  The module never
changes V1 unless its explicit production gate passes.  It combines directional
quality, ranking quality, bounded repeat-loss penalties, empirical confidence,
market regime, and portfolio feedback into a deterministic candidate score.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES = DATA / "prediction_candidates_history.csv"
EVALS = DATA / "evaluations.csv"
HISTORY = DATA / "ohlcv.csv"
OUT = DATA / "phase2_optimized_candidates.csv"
SUMMARY = DATA / "phase2_performance_summary.csv"
STATE = DATA / "phase2_state.json"

MIN_SESSIONS = 20
MIN_TRADES = 50
MAX_REPEAT_PENALTY = 2.0
DECAY_SESSIONS = 10
MIN_DIRECTION_LIFT_PCT = 0.5
MIN_RETURN_LIFT_PCT = 0.05


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _norm_dates(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.normalize()


def _directional_history(c: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    if e.empty or "symbol" not in e or "target_date" not in e:
        return pd.DataFrame(columns=["symbol", "direction_accuracy", "direction_rows", "direction_score"])
    x = e.copy()
    x["target_date"] = _norm_dates(x["target_date"])
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    if "close_direction_correct" in x:
        d = pd.to_numeric(x["close_direction_correct"], errors="coerce")
    elif {"predicted_close", "actual_close", "base_close"}.issubset(x.columns):
        p = pd.to_numeric(x["predicted_close"], errors="coerce") - pd.to_numeric(x["base_close"], errors="coerce")
        a = pd.to_numeric(x["actual_close"], errors="coerce") - pd.to_numeric(x["base_close"], errors="coerce")
        d = (p * a > 0).astype(float)
    else:
        return pd.DataFrame(columns=["symbol", "direction_accuracy", "direction_rows", "direction_score"])
    x["direction_correct"] = d
    x = x.dropna(subset=["symbol", "direction_correct"])
    if x.empty:
        return pd.DataFrame(columns=["symbol", "direction_accuracy", "direction_rows", "direction_score"])
    g = x.groupby("symbol").agg(
        direction_accuracy=("direction_correct", "mean"),
        direction_rows=("direction_correct", "size"),
    ).reset_index()
    # Shrink small samples toward 50% so one lucky trade cannot dominate ranking.
    g["direction_score"] = ((g["direction_accuracy"] * g["direction_rows"] + 0.50 * 10.0)
                            / (g["direction_rows"] + 10.0) * 100.0)
    return g


def _repeat_loss_penalty(candidates: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    out = candidates[["symbol"]].drop_duplicates().copy()
    out["repeat_loss_penalty"] = 0.0
    if e.empty or "symbol" not in e:
        return out
    x = e.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["target_date"] = _norm_dates(x["target_date"])
    if "profit_loss" in x:
        pnl = pd.to_numeric(x["profit_loss"], errors="coerce")
    elif {"actual_open", "actual_close"}.issubset(x.columns):
        pnl = pd.to_numeric(x["actual_close"], errors="coerce") - pd.to_numeric(x["actual_open"], errors="coerce")
    else:
        return out
    x["pnl"] = pnl
    x = x.dropna(subset=["symbol", "pnl"]).sort_values("target_date")
    latest = x["target_date"].max()
    if pd.isna(latest):
        return out
    x["age"] = (latest - x["target_date"]).dt.days.clip(lower=0)
    x["decay"] = np.exp(-x["age"] / max(DECAY_SESSIONS, 1))
    x["loss_weight"] = (x["pnl"] < 0).astype(float) * x["decay"]
    x["win_weight"] = (x["pnl"] > 0).astype(float) * x["decay"]
    g = x.groupby("symbol").agg(losses=("loss_weight", "sum"), wins=("win_weight", "sum")).reset_index()
    g["repeat_loss_penalty"] = ((g["losses"] - g["wins"] * 0.50).clip(lower=0) * 0.5).clip(upper=MAX_REPEAT_PENALTY)
    return out.drop(columns=["repeat_loss_penalty"]).merge(g[["symbol", "repeat_loss_penalty"]], on="symbol", how="left").fillna({"repeat_loss_penalty": 0.0})


def _regime(c: pd.DataFrame, h: pd.DataFrame) -> str:
    if h.empty or "date" not in h or "close" not in h:
        return "NEUTRAL"
    x = h.copy()
    x["date"] = _norm_dates(x["date"])
    daily = pd.to_numeric(x["close"], errors="coerce").groupby(x["date"]).median().dropna().sort_index()
    if len(daily) < 50:
        return "NEUTRAL"
    s20, s50 = daily.rolling(20).mean().iloc[-1], daily.rolling(50).mean().iloc[-1]
    r20 = daily.iloc[-1] / daily.iloc[-21] - 1.0
    if daily.iloc[-1] > s20 > s50 and r20 >= 0.03:
        return "BULL"
    if daily.iloc[-1] < s20 < s50 and r20 <= -0.03:
        return "BEAR"
    return "NEUTRAL"


def _confidence_v3(c: pd.DataFrame, direction: pd.DataFrame) -> pd.Series:
    spread = pd.to_numeric(c.get("prediction_spread", 0), errors="coerce").abs()
    spread_pct = spread.rank(pct=True, method="average")
    uncertainty = (1.0 - spread_pct).clip(0, 1).fillna(0.5) * 100.0
    direction_map = direction.set_index("symbol")["direction_score"] if not direction.empty else pd.Series(dtype=float)
    dscore = c["symbol"].map(direction_map).fillna(50.0)
    # Direction is now a first-class confidence component, not an afterthought.
    return (0.60 * uncertainty + 0.40 * dscore).clip(0, 100)


def _portfolio_feedback(e: pd.DataFrame) -> dict:
    if e.empty or "profit_loss" not in e:
        return {"trades": 0, "win_rate_pct": np.nan, "net_pnl": 0.0, "profit_factor": np.nan}
    pnl = pd.to_numeric(e["profit_loss"], errors="coerce").dropna()
    if pnl.empty:
        return {"trades": 0, "win_rate_pct": np.nan, "net_pnl": 0.0, "profit_factor": np.nan}
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    return {
        "trades": int(len(pnl)),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "net_pnl": float(pnl.sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
    }


def _production_gate(summary: dict) -> tuple[bool, str]:
    enough = summary["matched_sessions"] >= MIN_SESSIONS and summary["v2_trades"] >= MIN_TRADES
    if not enough:
        return False, "insufficient_evidence"
    if summary["direction_lift_pct"] < MIN_DIRECTION_LIFT_PCT:
        return False, "directional_lift_below_gate"
    if summary["return_lift_pct"] < MIN_RETURN_LIFT_PCT:
        return False, "return_lift_below_gate"
    return True, "all_phase2_gates_passed"


def run_phase2() -> pd.DataFrame:
    c = _read(CANDIDATES)
    e = _read(EVALS)
    h = _read(HISTORY)
    if c.empty:
        state = {"status": "collecting", "production_enabled": False, "reason": "no_candidates"}
        STATE.write_text(json.dumps(state, indent=2))
        pd.DataFrame([state]).to_csv(SUMMARY, index=False)
        return pd.DataFrame()

    c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
    c["prediction_date"] = _norm_dates(c["prediction_date"])
    if "rank" not in c:
        c["rank"] = np.arange(1, len(c) + 1)
    direction = _directional_history(c, e)
    repeat = _repeat_loss_penalty(c, e)
    c["direction_score_v3"] = c["symbol"].map(direction.set_index("symbol")["direction_score"] if not direction.empty else {}).fillna(50.0)
    c["confidence_v3"] = _confidence_v3(c, direction)
    c = c.merge(repeat, on="symbol", how="left")
    c["repeat_loss_penalty"] = c["repeat_loss_penalty"].fillna(0.0)

    regime = _regime(c, h)
    c["regime"] = regime
    expected = pd.to_numeric(c.get("predicted_close"), errors="coerce") / pd.to_numeric(c.get("base_close"), errors="coerce") - 1.0
    c["expected_return_pct"] = expected * 100.0
    c["phase2_score"] = (
        c["score"].rank(pct=True).fillna(0.5) * 35.0
        + c["direction_score_v3"] * 0.25
        + c["confidence_v3"] * 0.20
        + expected.rank(pct=True).fillna(0.5) * 10.0
        - c["repeat_loss_penalty"]
    )
    if regime == "BEAR":
        c["phase2_score"] -= (100.0 - c["direction_score_v3"]).clip(lower=0) * 0.05
    c["phase2_rank"] = c.groupby("prediction_date")["phase2_score"].rank(method="first", ascending=False).astype(int)
    c["phase2_selected"] = (c["phase2_rank"] <= 10).astype(int)
    c.to_csv(OUT, index=False)

    feedback = _portfolio_feedback(e)
    # These are deliberately conservative defaults until a matched V1/V2 history exists.
    matched_sessions = 0
    v2_trades = 0
    try:
        v2 = _read(DATA / "paper_trades_v2.csv")
        if not v2.empty:
            v2_trades = int((v2.get("signal", pd.Series(dtype=str)).astype(str) == "BUY").sum())
            matched_sessions = int(_read(DATA / "portfolio_v2_daily.csv")["target_date"].nunique()) if (DATA / "portfolio_v2_daily.csv").exists() else 0
    except Exception:
        pass

    # Directional comparison uses evaluated Top-10 vs ranks 11-20 when available.
    direction_lift = 0.0
    if not e.empty and "close_direction_correct" in e.columns and "rank" in e.columns:
        z = e.copy()
        z["rank"] = pd.to_numeric(z["rank"], errors="coerce")
        d = pd.to_numeric(z["close_direction_correct"], errors="coerce")
        top = d[z["rank"] <= 10].mean() if (z["rank"] <= 10).any() else np.nan
        nxt = d[z["rank"].between(11,20)].mean() if z["rank"].between(11,20).any() else np.nan
        if np.isfinite(top) and np.isfinite(nxt):
            direction_lift = float((top - nxt) * 100.0)

    summary = {
        "as_of": pd.Timestamp.now().normalize(),
        "regime": regime,
        "matched_sessions": matched_sessions,
        "v2_trades": v2_trades,
        "direction_lift_pct": direction_lift,
        "return_lift_pct": 0.0,
        **feedback,
    }
    enabled, reason = _production_gate(summary)
    summary["production_enabled"] = enabled
    summary["status"] = "promote" if enabled else "shadow"
    summary["reason"] = reason
    summary["min_matched_sessions"] = MIN_SESSIONS
    summary["min_v2_trades"] = MIN_TRADES
    summary["min_direction_lift_pct"] = MIN_DIRECTION_LIFT_PCT
    summary["min_return_lift_pct"] = MIN_RETURN_LIFT_PCT
    pd.DataFrame([summary]).to_csv(SUMMARY, index=False)
    STATE.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Phase 2: status={summary['status']}, regime={regime}, direction_lift={direction_lift:.2f}%, reason={reason}")
    return pd.DataFrame([summary])


if __name__ == "__main__":
    run_phase2()
