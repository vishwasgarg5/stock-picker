from __future__ import annotations

"""Build a compact Phase-2 performance dashboard from persisted evidence."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "phase2_dashboard.csv"
JSON_OUT = DATA / "phase2_dashboard.json"


def _read(name: str) -> pd.DataFrame:
    p = DATA / name
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _last(df: pd.DataFrame, col: str, default=np.nan):
    if df.empty or col not in df.columns:
        return default
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else default


def build_dashboard() -> dict:
    p2 = _read("phase2_performance_summary.csv")
    hist = _read("performance_history.csv")
    ab = _read("strategy_ab_comparison.csv")
    direction = _read("directional_model_validation_summary.csv")
    v2 = _read("paper_trades_v2.csv")
    v2daily = _read("portfolio_v2_daily.csv")
    governance = _read("model_governance_summary.csv")
    phase4 = _read("phase4_performance_summary.csv")
    state = {}
    state_path = DATA / "phase2_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}

    latest = p2.iloc[-1].to_dict() if not p2.empty else {}
    dashboard = {
        "as_of": str(latest.get("as_of", "")),
        "production_strategy": "V1",
        "phase2_status": latest.get("status", state.get("status", "shadow")),
        "phase2_reason": latest.get("reason", state.get("reason", "insufficient_evidence")),
        "regime": latest.get("regime", "NEUTRAL"),
        "v2_matched_sessions": int(pd.to_datetime(v2daily.get("target_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.normalize().nunique()) if not v2daily.empty and "target_date" in v2daily.columns else int(latest.get("matched_sessions", 0) or 0),
        "v2_trades": int(v2["signal"].astype(str).str.upper().eq("BUY").sum()) if not v2.empty and "signal" in v2.columns else int(latest.get("v2_trades", 0) or 0),
        "shadow_direction_lift_pct": float(latest.get("direction_lift_pct", 0.0) or 0.0),
        "shadow_return_lift_pct": float(latest.get("return_lift_pct", 0.0) or 0.0),
        "latest_close_mape_pct": _last(hist, "close_mape_pct"),
        "latest_baseline_close_mape_pct": _last(hist, "baseline_close_mape_pct"),
        "latest_direction_accuracy_pct": _last(hist, "direction_accuracy_pct"),
        "rolling5_close_improvement_pct": _last(hist, "close_improvement_5s_pct"),
        "v2_ab_return_lift_pct": _last(ab, "return_lift_pct"),
        "v2_ab_session_win_rate_pct": _last(ab, "session_win_rate_pct"),
        "v2_ab_drawdown_gate": bool(str(ab.iloc[-1].get("drawdown_gate", "False")).lower() == "true") if not ab.empty else False,
        "directional_challenger_accuracy_pct": _last(direction, "mean_accuracy_pct"),
        "directional_challenger_lift_pct": _last(direction, "mean_accuracy_lift_pct"),
        "directional_challenger_ready": bool(str(direction.iloc[-1].get("production_ready", "False")).lower() == "true") if not direction.empty else False,
        "return_lift_ci_low_pct": _last(ab, "return_lift_ci_low_pct", -np.inf),
        "promotion_safe": bool(
            (not governance.empty and str(governance.iloc[-1].get("safety_status","FAIL")).upper() == "PASS")
            and (not governance.empty and float(pd.to_numeric(governance.iloc[-1].get("bootstrap_ci_low_pct"), errors="coerce")) >= 0.0)
            and (not direction.empty and str(direction.iloc[-1].get("production_ready","False")).lower() == "true")
            and (not ab.empty and float(pd.to_numeric(ab.iloc[-1].get("common_sessions"), errors="coerce")) >= 20)
            and (not ab.empty and float(pd.to_numeric(ab.iloc[-1].get("v2_trades"), errors="coerce")) >= 50)
        ),
        "bootstrap_ci_low_pct": _last(governance, "bootstrap_ci_low_pct", -np.inf),
        "bootstrap_ci_high_pct": _last(governance, "bootstrap_ci_high_pct", np.inf),
        "feature_drift_count": _last(governance, "feature_drift_count", np.nan),
        "drift_status": str(governance.iloc[-1].get("drift_status","UNKNOWN")) if not governance.empty else "UNKNOWN",
        "model_drift_status": str(governance.iloc[-1].get("model_drift_status","UNKNOWN")) if not governance.empty else "UNKNOWN",
        "recent_direction_accuracy_pct": _last(governance, "recent_direction_accuracy_pct", np.nan),
        "unstable_stocks": _last(governance, "unstable_stocks", np.nan),
        "stable_features": _last(governance, "stable_features", np.nan),
        "v2_worst_trade_pnl": _last(governance, "worst_v2_trade_pnl", np.nan),
        "v2_max_consecutive_losses": _last(governance, "max_consecutive_v2_losses", np.nan),
        "v2_profit_factor": _last(governance, "v2_profit_factor", np.nan),
        "phase4_regime": str(phase4.iloc[-1].get("regime", "NEUTRAL")) if not phase4.empty else "NEUTRAL",
        "phase4_confidence_floor_percentile": _last(phase4, "confidence_floor_percentile", np.nan),
        "phase4_min_expected_return_pct": _last(phase4, "min_expected_return_pct", np.nan),
        "phase4_cooldown_symbols": _last(phase4, "cooldown_symbols", np.nan),
        "phase4_stable_features": _last(phase4, "stable_features", np.nan),
        "phase4_bootstrap_ci_low_pct": _last(phase4, "bootstrap_ci_low_pct", -np.inf),
        "phase4_bootstrap_ci_high_pct": _last(phase4, "bootstrap_ci_high_pct", np.inf),
        "phase4_safety_status": str(phase4.iloc[-1].get("safety_status", "UNKNOWN")) if not phase4.empty else "UNKNOWN",
    }
    dashboard["model_beating_baseline"] = bool(
        np.isfinite(dashboard["latest_close_mape_pct"])
        and np.isfinite(dashboard["latest_baseline_close_mape_pct"])
        and dashboard["latest_close_mape_pct"] < dashboard["latest_baseline_close_mape_pct"]
    )
    dashboard["directional_quality_warning"] = bool(
        np.isfinite(dashboard["latest_direction_accuracy_pct"])
        and dashboard["latest_direction_accuracy_pct"] < 50.0
    )

    pd.DataFrame([dashboard]).to_csv(OUT, index=False)
    JSON_OUT.write_text(json.dumps(dashboard, indent=2, default=str))
    print(json.dumps(dashboard, indent=2))
    return dashboard


if __name__ == "__main__":
    build_dashboard()
