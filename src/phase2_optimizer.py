from __future__ import annotations

"""Phase 2 performance optimizer.

Runs all eight performance improvements in shadow mode.  V1 remains the
production champion until independent matched-session/trade evidence passes.
The optimizer is deliberately conservative: directional quality and realized
trade returns are evaluated separately from price-magnitude MAPE.
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
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
    # Bayesian-style shrinkage toward 50% for small samples.
    g["direction_score"] = (
        (g["direction_accuracy"] * g["direction_rows"] + 0.50 * 10.0)
        / (g["direction_rows"] + 10.0) * 100.0
    )
    return g


def _repeat_loss_penalty(candidates: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    out = candidates[["symbol"]].drop_duplicates().copy()
    out["repeat_loss_penalty"] = 0.0
    if e.empty or "symbol" not in e or "target_date" not in e:
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
    x = x.dropna(subset=["symbol", "pnl", "target_date"]).sort_values("target_date")
    if x.empty:
        return out
    latest = x["target_date"].max()
    x["age"] = (latest - x["target_date"]).dt.days.clip(lower=0)
    x["decay"] = np.exp(-x["age"] / max(DECAY_SESSIONS, 1))
    x["loss_weight"] = (x["pnl"] < 0).astype(float) * x["decay"]
    x["win_weight"] = (x["pnl"] > 0).astype(float) * x["decay"]
    g = x.groupby("symbol").agg(losses=("loss_weight", "sum"), wins=("win_weight", "sum")).reset_index()
    g["repeat_loss_penalty"] = ((g["losses"] - g["wins"] * 0.50).clip(lower=0) * 0.5).clip(upper=MAX_REPEAT_PENALTY)
    return out.drop(columns=["repeat_loss_penalty"]).merge(
        g[["symbol", "repeat_loss_penalty"]], on="symbol", how="left"
    ).fillna({"repeat_loss_penalty": 0.0})


def _regime(c: pd.DataFrame, h: pd.DataFrame) -> str:
    if h.empty or not {"date", "close"}.issubset(h.columns):
        return "NEUTRAL"
    x = h.copy()
    x["date"] = _norm_dates(x["date"])
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    daily = x.dropna(subset=["date", "close"]).groupby("date")["close"].median().sort_index()
    if len(daily) < 50:
        return "NEUTRAL"
    s20 = daily.rolling(20).mean().iloc[-1]
    s50 = daily.rolling(50).mean().iloc[-1]
    r20 = daily.iloc[-1] / daily.iloc[-21] - 1.0
    if daily.iloc[-1] > s20 > s50 and r20 >= 0.03:
        return "BULL"
    if daily.iloc[-1] < s20 < s50 and r20 <= -0.03:
        return "BEAR"
    return "NEUTRAL"


def _confidence_v3(c: pd.DataFrame, direction: pd.DataFrame) -> pd.Series:
    spread = pd.to_numeric(c.get("prediction_spread", pd.Series(np.nan, index=c.index)), errors="coerce").abs()
    # Smaller prediction spread = lower uncertainty. Rank only valid spreads.
    spread_pct = spread.rank(pct=True, method="average")
    uncertainty = (1.0 - spread_pct).clip(0, 1).fillna(0.5) * 100.0
    if direction.empty:
        dscore = pd.Series(50.0, index=c.index)
    else:
        dmap = direction.set_index("symbol")["direction_score"]
        dscore = c["symbol"].map(dmap).fillna(50.0)
    return (0.60 * uncertainty + 0.40 * dscore).clip(0, 100)


def _portfolio_feedback(trades: pd.DataFrame) -> dict:
    if trades.empty or "profit_loss" not in trades.columns:
        return {"trades": 0, "win_rate_pct": np.nan, "net_pnl": 0.0, "profit_factor": np.nan}
    x = trades.copy()
    if "signal" in x.columns:
        x = x[x["signal"].astype(str).str.upper().eq("BUY")]
    pnl = pd.to_numeric(x["profit_loss"], errors="coerce").dropna()
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


def _shadow_selection_metrics(c: pd.DataFrame, e: pd.DataFrame) -> tuple[float, float, int]:
    """Compare Phase-2 Top-10 against the original rank Top-10 on completed sessions."""
    if c.empty or e.empty or "rank" not in c.columns:
        return 0.0, 0.0, 0
    keys = [k for k in ["prediction_date", "symbol"] if k in c.columns and k in e.columns]
    if keys != ["prediction_date", "symbol"]:
        return 0.0, 0.0, 0
    cols = ["prediction_date", "symbol", "phase2_selected", "phase2_rank"]
    cc = c[cols].copy()
    cc["prediction_date"] = _norm_dates(cc["prediction_date"])
    ee = e.copy()
    ee["prediction_date"] = _norm_dates(ee["prediction_date"])
    ee["symbol"] = ee["symbol"].astype(str).str.upper().str.strip()
    cc["symbol"] = cc["symbol"].astype(str).str.upper().str.strip()
    m = ee.merge(cc, on=["prediction_date", "symbol"], how="inner")
    if m.empty:
        return 0.0, 0.0, 0
    direction = pd.to_numeric(m.get("close_direction_correct"), errors="coerce")
    m["actual_return_pct"] = np.nan
    if {"actual_open", "actual_close"}.issubset(m.columns):
        op = pd.to_numeric(m["actual_open"], errors="coerce")
        cl = pd.to_numeric(m["actual_close"], errors="coerce")
        m["actual_return_pct"] = np.where(op.ne(0), (cl / op - 1.0) * 100.0, np.nan)
    phase = m[m["phase2_selected"].eq(1)]
    base = m[pd.to_numeric(m["rank"], errors="coerce") <= 10]
    if phase.empty or base.empty:
        return 0.0, 0.0, int(m["prediction_date"].nunique())
    pdir = direction.loc[phase.index].mean()
    bdir = direction.loc[base.index].mean()
    pret = pd.to_numeric(phase["actual_return_pct"], errors="coerce").mean()
    bret = pd.to_numeric(base["actual_return_pct"], errors="coerce").mean()
    direction_lift = float((pdir - bdir) * 100.0) if np.isfinite(pdir) and np.isfinite(bdir) else 0.0
    return_lift = float(pret - bret) if np.isfinite(pret) and np.isfinite(bret) else 0.0
    return direction_lift, return_lift, int(m["prediction_date"].nunique())


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
    c = c.dropna(subset=["symbol", "prediction_date"]).copy()
    if c.empty:
        state = {"status": "collecting", "production_enabled": False, "reason": "no_valid_candidates"}
        STATE.write_text(json.dumps(state, indent=2))
        pd.DataFrame([state]).to_csv(SUMMARY, index=False)
        return pd.DataFrame()

    for col in ["score", "predicted_close", "base_close", "prediction_spread", "rank"]:
        if col in c:
            c[col] = pd.to_numeric(c[col], errors="coerce")
    c["score"] = c["score"].fillna(50.0) if "score" in c else 50.0
    if "rank" not in c:
        c["rank"] = c.groupby("prediction_date").cumcount() + 1

    direction = _directional_history(c, e)
    repeat = _repeat_loss_penalty(c, e)
    dmap = direction.set_index("symbol")["direction_score"] if not direction.empty else pd.Series(dtype=float)
    historical_direction = c["symbol"].map(dmap).fillna(50.0)
    model_direction = pd.to_numeric(c.get("direction_score_model", pd.Series(np.nan, index=c.index)), errors="coerce")
    c["direction_score_v3"] = (0.60 * historical_direction + 0.40 * model_direction.fillna(historical_direction)).clip(0, 100)
    c["confidence_v3"] = _confidence_v3(c, direction)
    c = c.merge(repeat, on="symbol", how="left")
    c["repeat_loss_penalty"] = pd.to_numeric(c["repeat_loss_penalty"], errors="coerce").fillna(0.0)

    regime = _regime(c, h)
    c["regime"] = regime
    predicted_close = pd.to_numeric(c.get("predicted_close", pd.Series(np.nan, index=c.index)), errors="coerce")
    base_close = pd.to_numeric(c.get("base_close", pd.Series(np.nan, index=c.index)), errors="coerce")
    expected = predicted_close.div(base_close.replace(0, np.nan)) - 1.0
    c["expected_return_pct"] = expected.replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0

    # Rank components independently so no single raw model score dominates.
    rank_score = c.groupby("prediction_date")["score"].rank(pct=True, method="average").fillna(0.5) * 35.0
    return_rank = c.groupby("prediction_date")["expected_return_pct"].rank(pct=True, method="average").fillna(0.5) * 10.0
    direction_component = c["direction_score_v3"] * 0.25
    confidence_component = c["confidence_v3"] * 0.20
    c["phase2_score"] = rank_score + direction_component + confidence_component + return_rank - c["repeat_loss_penalty"]
    if regime == "BEAR":
        c["phase2_score"] -= (100.0 - c["direction_score_v3"]).clip(lower=0) * 0.05
    c["phase2_score"] = pd.to_numeric(c["phase2_score"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(50.0)
    c["phase2_score"] = c["phase2_score"].clip(lower=-1e6, upper=1e6)
    c["phase2_rank"] = (
        c.groupby("prediction_date")["phase2_score"].rank(method="first", ascending=False)
        .fillna(c.groupby("prediction_date")["phase2_score"].transform("size").fillna(1) + 1)
        .astype("int64")
    )
    c["phase2_selected"] = (c["phase2_rank"] <= 10).astype(int)
    c["phase2_top5"] = (c["phase2_rank"] <= 5).astype(int)
    c.to_csv(OUT, index=False)

    feedback = _portfolio_feedback(_read(DATA / "paper_trades.csv"))
    direction_lift, return_lift, matched_eval_sessions = _shadow_selection_metrics(c, e)

    # Independent V2 evidence is read from the actual paper-trading history.
    v2 = _read(DATA / "paper_trades_v2.csv")
    v2daily = _read(DATA / "portfolio_v2_daily.csv")
    if not v2.empty and "signal" in v2.columns:
        v2_trades = int(v2["signal"].astype(str).str.upper().eq("BUY").sum())
    else:
        v2_trades = 0
    if not v2daily.empty and "target_date" in v2daily.columns:
        matched_sessions = int(_norm_dates(v2daily["target_date"]).dropna().nunique())
    else:
        matched_sessions = 0

    summary = {
        "as_of": pd.Timestamp.now().normalize(),
        "regime": regime,
        "matched_sessions": matched_sessions,
        "v2_trades": v2_trades,
        "evaluated_shadow_sessions": matched_eval_sessions,
        "direction_lift_pct": direction_lift,
        "direction_model_available_pct": float(model_direction.notna().mean() * 100.0) if len(model_direction) else 0.0,
        "return_lift_pct": return_lift,
        **feedback,
    }
    enabled, reason = _production_gate(summary)
    summary.update({
        "production_enabled": enabled,
        "status": "promote" if enabled else "shadow",
        "reason": reason,
        "min_matched_sessions": MIN_SESSIONS,
        "min_v2_trades": MIN_TRADES,
        "min_direction_lift_pct": MIN_DIRECTION_LIFT_PCT,
        "min_return_lift_pct": MIN_RETURN_LIFT_PCT,
    })
    pd.DataFrame([summary]).to_csv(SUMMARY, index=False)
    STATE.write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"Phase 2: status={summary['status']}, regime={regime}, "
        f"shadow_direction_lift={direction_lift:.2f}%, shadow_return_lift={return_lift:.2f}%, "
        f"V2_sessions={matched_sessions}, V2_trades={v2_trades}, reason={reason}"
    )
    return pd.DataFrame([summary])


if __name__ == "__main__":
    run_phase2()
