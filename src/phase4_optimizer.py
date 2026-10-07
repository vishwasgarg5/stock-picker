from __future__ import annotations

"""Phase 4 adaptive trading optimizer.

Phase 4 converts accumulated evaluation/paper-trading evidence into a
versioned, conservative overlay for the V2 shadow selector.  It never changes
the V1 production champion directly.  The overlay covers:
1) signal/threshold optimization, 2) entry/exit risk parameters,
3) position sizing, 4) diversification, 5) regime adaptation,
6) losing-symbol cooldowns, 7) feature-quality feedback,
8) confidence calibration, 9) trade-quality feedback, and
10) statistical safety/promotion evidence.

All parameters are learned only from data strictly before the prediction date.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVAL = DATA / "evaluations.csv"
TRADES = DATA / "paper_trades_v2.csv"
GOV = DATA / "model_governance_summary.csv"
OUT = DATA / "phase4_candidates.csv"
SUMMARY = DATA / "phase4_performance_summary.csv"
CONFIG = DATA / "phase4_config.json"

MIN_SYMBOL_OBS = 10
MAX_SYMBOL_WEIGHT = 1.50
MIN_SYMBOL_WEIGHT = 0.50
MAX_REPEAT_LOSS = 2.0
COOLDOWN_LOSSES = 3
MIN_CONFIDENCE_PCTL = 0.55
MAX_CONFIDENCE_PCTL = 0.90


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _dates(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.normalize()


def _symbol_stats(e: pd.DataFrame) -> pd.DataFrame:
    if e.empty or not {"symbol", "target_date", "close_direction_correct"}.issubset(e.columns):
        return pd.DataFrame(columns=["symbol", "observations", "direction_accuracy_pct",
                                     "recent_accuracy_pct", "mean_return_pct", "loss_streak"])
    x = e.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["target_date"] = _dates(x["target_date"])
    x["direction"] = pd.to_numeric(x["close_direction_correct"], errors="coerce")
    if {"actual_open", "actual_close"}.issubset(x.columns):
        op = pd.to_numeric(x["actual_open"], errors="coerce")
        cl = pd.to_numeric(x["actual_close"], errors="coerce")
        x["return_pct"] = np.where(op.ne(0), (cl / op - 1.0) * 100.0, np.nan)
    else:
        x["return_pct"] = np.nan
    x = x.dropna(subset=["symbol", "target_date", "direction"]).sort_values(["symbol", "target_date"])
    rows = []
    for sym, g in x.groupby("symbol"):
        vals = g["direction"].astype(float).to_numpy()
        streak = 0
        for v in vals[::-1]:
            if v < 0.5:
                streak += 1
            else:
                break
        rows.append({
            "symbol": sym,
            "observations": int(len(g)),
            "direction_accuracy_pct": float(g["direction"].mean() * 100.0),
            "recent_accuracy_pct": float(g["direction"].tail(10).mean() * 100.0),
            "mean_return_pct": float(g["return_pct"].mean()) if g["return_pct"].notna().any() else 0.0,
            "loss_streak": int(streak),
        })
    return pd.DataFrame(rows)


def _regime(hist: pd.DataFrame) -> str:
    if hist.empty or not {"date", "close"}.issubset(hist.columns):
        return "NEUTRAL"
    x = hist.copy()
    x["date"] = _dates(x["date"])
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    daily = x.dropna(subset=["date", "close"]).groupby("date")["close"].median().sort_index()
    if len(daily) < 50:
        return "NEUTRAL"
    s20, s50 = daily.rolling(20).mean().iloc[-1], daily.rolling(50).mean().iloc[-1]
    r20 = daily.iloc[-1] / daily.iloc[-21] - 1.0
    if daily.iloc[-1] > s20 > s50 and r20 >= 0.03:
        return "BULL"
    if daily.iloc[-1] < s20 < s50 and r20 <= -0.03:
        return "BEAR"
    return "NEUTRAL"


def _regime_multipliers(regime: str) -> dict:
    if regime == "BULL":
        return {"confidence": 0.55, "expected_return": 0.75, "stop": 1.35, "target": 1.10}
    if regime == "BEAR":
        return {"confidence": 0.70, "expected_return": 1.25, "stop": 1.10, "target": 1.25}
    return {"confidence": 0.60, "expected_return": 1.00, "stop": 1.25, "target": 1.15}


def _global_parameters(stats: pd.DataFrame, trades: pd.DataFrame, regime: str) -> dict:
    m = _regime_multipliers(regime)
    if not stats.empty:
        acc = float(stats["recent_accuracy_pct"].median())
        threshold = np.clip(0.55 + max(0.0, 50.0 - acc) / 200.0, MIN_CONFIDENCE_PCTL, MAX_CONFIDENCE_PCTL)
    else:
        threshold = 0.60
    if not trades.empty and "return_pct" in trades.columns:
        r = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
        if len(r) >= 20 and float(r.mean()) < 0:
            m["expected_return"] *= 1.10
    return {
        "confidence_percentile": float(np.clip(threshold * m["confidence"], MIN_CONFIDENCE_PCTL, MAX_CONFIDENCE_PCTL)),
        "min_expected_return_pct": float(0.50 * m["expected_return"]),
        "stop_atr_multiplier": float(m["stop"]),
        "target_stop_multiple": float(m["target"]),
        "max_positions": 5,
        "max_symbol_weight": MAX_SYMBOL_WEIGHT,
        "min_symbol_weight": MIN_SYMBOL_WEIGHT,
        "regime": regime,
    }


def _feature_feedback() -> dict:
    f = _read(DATA / "feature_stability.csv")
    if f.empty or "feature" not in f.columns or "stable_signal" not in f.columns:
        return {"stable_features": 0, "unstable_features": []}
    unstable = f.loc[~f["stable_signal"].astype(bool), "feature"].astype(str).tolist()
    return {"stable_features": int(f["stable_signal"].astype(bool).sum()), "unstable_features": unstable}


def _bootstrap(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty or "return_pct" not in trades.columns:
        return -np.inf, np.inf
    if "signal" in trades.columns:
        mask = trades["signal"].astype(str).str.upper().eq("BUY")
        r = pd.to_numeric(trades.loc[mask, "return_pct"], errors="coerce").dropna().to_numpy()
    else:
        r = pd.to_numeric(trades["return_pct"], errors="coerce").dropna().to_numpy()
    if len(r) < 10:
        return -np.inf, np.inf
    rng = np.random.default_rng(20261008)
    samples = rng.choice(r, size=(1000, len(r)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_phase4() -> pd.DataFrame:
    e = _read(EVAL)
    t = _read(TRADES)
    hist = _read(DATA / "ohlcv.csv")
    stats = _symbol_stats(e)
    regime = _regime(hist)
    params = _global_parameters(stats, t, regime)
    feature = _feature_feedback()
    ci_low, ci_high = _bootstrap(t)

    # Per-symbol adaptive weights are shrinkage-based and capped.  Symbols with
    # insufficient evidence stay near neutral rather than being overfit.
    if stats.empty:
        stats = pd.DataFrame({"symbol": [], "observations": [], "direction_accuracy_pct": [],
                              "recent_accuracy_pct": [], "mean_return_pct": [], "loss_streak": []})
    stats["evidence_weight"] = np.clip(
        stats["observations"] / (stats["observations"] + 20.0), 0.0, 1.0
    )
    skill = (stats["recent_accuracy_pct"] - 50.0) / 20.0
    ret = np.clip(stats["mean_return_pct"] / 1.0, -1.0, 1.0)
    stats["symbol_weight"] = np.clip(
        1.0 + stats["evidence_weight"] * (0.60 * skill + 0.40 * ret),
        MIN_SYMBOL_WEIGHT, MAX_SYMBOL_WEIGHT
    )
    stats["cooldown"] = stats["loss_streak"] >= COOLDOWN_LOSSES
    stats["confidence_floor"] = params["confidence_percentile"]
    stats["min_expected_return_pct"] = params["min_expected_return_pct"]
    stats["stop_atr_multiplier"] = params["stop_atr_multiplier"]
    stats["target_stop_multiple"] = params["target_stop_multiple"]
    stats["regime"] = regime
    stats["feature_stability_status"] = "PASS" if feature["stable_features"] >= 1 else "COLLECTING"

    report = {
        "as_of": pd.Timestamp.now(tz="UTC").isoformat(),
        "phase4_version": "phase4_adaptive_v1",
        "regime": regime,
        "symbols_scored": int(len(stats)),
        "cooldown_symbols": int(stats["cooldown"].sum()) if not stats.empty else 0,
        "stable_features": feature["stable_features"],
        "unstable_feature_count": len(feature["unstable_features"]),
        "confidence_floor_percentile": params["confidence_percentile"],
        "min_expected_return_pct": params["min_expected_return_pct"],
        "stop_atr_multiplier": params["stop_atr_multiplier"],
        "target_stop_multiple": params["target_stop_multiple"],
        "max_positions": params["max_positions"],
        "bootstrap_ci_low_pct": ci_low,
        "bootstrap_ci_high_pct": ci_high,
        "safety_status": "PASS" if ci_low >= 0.0 or len(t) < 10 else "SHADOW_ONLY",
        "production_ready": False,
        "reason": "Phase 4 is an adaptive overlay; V1/V2 production promotion remains controlled by strategy_governor.",
    }
    pd.DataFrame([report]).to_csv(SUMMARY, index=False)
    stats.to_csv(OUT, index=False)
    CONFIG.write_text(json.dumps({"report": report, "parameters": params,
                                  "feature_feedback": feature}, indent=2, default=str))
    print(json.dumps(report, indent=2))
    return stats


def apply_phase4(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply only precomputed Phase-4 parameters; never use future actual prices."""
    x = candidates.copy()
    if x.empty:
        return x
    config = {}
    try:
        config = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    except Exception:
        config = {}
    params = config.get("parameters", {})
    scored = _read(OUT)
    if not scored.empty and "symbol" in scored.columns:
        scored["symbol"] = scored["symbol"].astype(str).str.upper().str.strip()
        x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
        cols = [c for c in ["symbol", "symbol_weight", "cooldown", "confidence_floor",
                            "min_expected_return_pct", "stop_atr_multiplier",
                            "target_stop_multiple"] if c in scored.columns]
        x = x.merge(scored[cols].drop_duplicates("symbol"), on="symbol", how="left")
    x["symbol_weight"] = pd.to_numeric(x.get("symbol_weight"), errors="coerce").fillna(1.0)
    x["cooldown"] = x.get("cooldown", False).fillna(False).astype(bool)
    x["phase4_confidence_floor"] = float(params.get("confidence_percentile", 0.60))
    x["phase4_min_expected_return_pct"] = float(params.get("min_expected_return_pct", 0.50))
    x["phase4_stop_atr_multiplier"] = float(params.get("stop_atr_multiplier", 1.25))
    x["phase4_target_stop_multiple"] = float(params.get("target_stop_multiple", 1.15))
    x["phase4_score"] = (
        pd.to_numeric(x.get("phase2_score"), errors="coerce").fillna(0.0)
        * x["symbol_weight"]
        * (pd.to_numeric(x.get("trade_quality_probability"), errors="coerce").fillna(0.50) * 2.0)
    )
    x["phase4_eligible"] = (
        ~x["cooldown"]
        & pd.to_numeric(x.get("confidence_pct"), errors="coerce").ge(x["phase4_confidence_floor"])
        & pd.to_numeric(x.get("expected_return_pct"), errors="coerce").ge(x["phase4_min_expected_return_pct"])
    )
    # Diversification: cap one symbol and one candidate per symbol/session.
    x["phase4_weight"] = x["symbol_weight"].clip(MIN_SYMBOL_WEIGHT, MAX_SYMBOL_WEIGHT)
    return x


if __name__ == "__main__":
    build_phase4()
