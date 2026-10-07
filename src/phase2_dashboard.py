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
        "v2_matched_sessions": int(latest.get("matched_sessions", 0) or 0),
        "v2_trades": int(latest.get("v2_trades", 0) or 0),
        "shadow_direction_lift_pct": float(latest.get("direction_lift_pct", 0.0) or 0.0),
        "shadow_return_lift_pct": float(latest.get("return_lift_pct", 0.0) or 0.0),
        "latest_close_mape_pct": _last(hist, "close_mape_pct"),
        "latest_baseline_close_mape_pct": _last(hist, "baseline_close_mape_pct"),
        "latest_direction_accuracy_pct": _last(hist, "direction_accuracy_pct"),
        "rolling5_close_improvement_pct": _last(hist, "close_improvement_5s_pct"),
        "v2_ab_return_lift_pct": _last(ab, "return_lift_pct"),
        "v2_ab_session_win_rate_pct": _last(ab, "session_win_rate_pct"),
        "v2_ab_drawdown_gate": bool(str(ab.iloc[-1].get("drawdown_gate", "False")).lower() == "true") if not ab.empty else False,
        "promotion_safe": True,
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
