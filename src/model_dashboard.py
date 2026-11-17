"""Build a live, evidence-backed static dashboard from persisted model artifacts."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "dashboard_data.json"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _csv(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) and math.isfinite(float(parsed)) else None


def _latest_date(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    return dates.max().date().isoformat() if not dates.empty else None


def _status(value: Any, fallback: str = "NOT_AVAILABLE") -> str:
    return str(value).strip().upper() if value not in (None, "") else fallback


def build_dashboard() -> dict[str, Any]:
    predictions = _csv("predictions.csv")
    evaluations = _csv("evaluations.csv")
    trades_v1 = _csv("paper_trades.csv")
    trades_v2 = _csv("paper_trades_v2.csv")
    coverage = _json(DATA / "historical_coverage_audit.json")
    forecast = _json(DATA / "phase110_forecast_accuracy_summary.json")
    replay = _json(DATA / "phase111_chronological_replay_summary.json")
    pnl = _json(DATA / "phase107_pnl_reconciliation_summary.json")
    funnel = _json(DATA / "phase109_selection_funnel_summary.json")
    promotion = _json(DATA / "phase114_promotion_readiness_summary.json")
    market = _json(DATA / "phase112_market_intelligence_summary.json")
    governance = _json(DATA / "model_governance.json")
    calendar = _json(DATA / "nse_market_calendar.json")
    phase2 = _json(DATA / "phase2_dashboard.json")
    performance = _csv("performance_audit.csv")
    progress = _csv("model_progress_audit.csv")

    latest_prediction_date = _latest_date(predictions, "prediction_date")
    latest_predictions: list[dict[str, Any]] = []
    if latest_prediction_date and {"prediction_date", "symbol"}.issubset(predictions.columns):
        rows = predictions.loc[
            pd.to_datetime(predictions["prediction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            == latest_prediction_date
        ].copy()
        if "rank" in rows:
            rows["_rank"] = pd.to_numeric(rows["rank"], errors="coerce")
            rows = rows.sort_values(["_rank", "symbol"], na_position="last")
        columns = [
            "rank", "symbol", "target_date", "base_close", "predicted_close",
            "score", "confidence_score", "direction_probability", "risk_level",
            "selection_explanation",
        ]
        latest_predictions = _clean(rows[[c for c in columns if c in rows]].head(10).to_dict("records"))

    def trade_summary(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"rows": 0, "buy_rows": 0, "skip_rows": 0, "net_pnl": None, "win_rate_pct": None}
        signal = frame.get("signal", pd.Series("", index=frame.index)).astype(str).str.upper().str.strip()
        buys = frame.loc[signal.eq("BUY")]
        pnl_col = "profit_loss" if "profit_loss" in buys else None
        pnl_values = pd.to_numeric(buys[pnl_col], errors="coerce").dropna() if pnl_col else pd.Series(dtype=float)
        return {
            "rows": int(len(frame)),
            "buy_rows": int(len(buys)),
            "skip_rows": int(signal.eq("SKIP").sum()),
            "net_pnl": float(pnl_values.sum()) if not pnl_values.empty else None,
            "win_rate_pct": float((pnl_values > 0).mean() * 100) if not pnl_values.empty else None,
            "last_target_date": _latest_date(frame, "target_date"),
        }

    metric_close = forecast.get("metrics", {}).get("close", {})
    index_latest = market.get("latest_index_session") or market.get("reference_ohlcv_session")
    index_stale = bool(market.get("stale_index_rows", 0) or market.get("future_index_rows", 0) or market.get("duplicate_index_rows", 0))
    calendar_updated = calendar.get("updated_at_utc")
    calendar_status = "PASS" if calendar_updated else "REVIEW"
    coverage_status = _status(coverage.get("status"), "NOT_AVAILABLE")
    promotion_status = _status(promotion.get("status"), "NOT_AVAILABLE")
    market_status = "FAIL" if index_stale else ("PASS" if market.get("index_data_available") and index_latest else "REVIEW")

    performance_findings: list[str] = []
    if not performance.empty and "findings" in performance:
        performance_findings = [str(x) for x in performance["findings"].dropna().tail(3).tolist()]
    progress_counts: dict[str, int] = {}
    if not progress.empty and "status" in progress:
        progress_counts = {str(k).upper(): int(v) for k, v in progress["status"].astype(str).str.upper().value_counts().items()}

    return _clean({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "repository data artifacts; values are not inferred",
        "production": {"champion": forecast.get("production_champion", "V1"), "challenger_mode": forecast.get("v2_mode", "SHADOW_ONLY"), "promotion_status": promotion_status, "automatic_promotion": bool(promotion.get("automatic_promotion", False)), "production_changes_authorized": bool(promotion.get("production_changes_authorized", False))},
        "predictions": {"prediction_date": latest_prediction_date, "target_date": _latest_date(predictions.loc[pd.to_datetime(predictions.get("prediction_date"), errors="coerce").dt.strftime("%Y-%m-%d").eq(latest_prediction_date)] if latest_prediction_date else pd.DataFrame(), "target_date") if latest_prediction_date else None, "count": len(latest_predictions), "top10": latest_predictions},
        "forecast": {"status": _status(forecast.get("status")), "evaluated_rows": forecast.get("matched_rows"), "close_mape_pct": metric_close.get("mape_pct"), "close_mae": metric_close.get("mae"), "directional_accuracy_pct": forecast.get("close_directional_accuracy_pct"), "chronological_status": _status(replay.get("status")), "v1": replay.get("v1", {}), "v2": replay.get("v2", {})},
        "paper_trading": {"v1": trade_summary(trades_v1), "v2": trade_summary(trades_v2), "v2_funnel": {"status": _status(funnel.get("status")), "buy_rows": funnel.get("buy_rows"), "skip_rows": funnel.get("skip_rows"), "skip_rows_without_reason": funnel.get("skip_rows_without_reason"), "top_rejection_reasons": funnel.get("skip_reason_counts", {})}},
        "data_health": {"historical_coverage": {"status": coverage_status, "universe_symbols": coverage.get("universe_symbols"), "symbols_with_history": coverage.get("symbols_with_history"), "training_ready_50_sessions": coverage.get("symbols_training_ready_50_sessions"), "history_start": coverage.get("history_start"), "history_end": coverage.get("history_end"), "valid_rows": coverage.get("total_valid_rows"), "median_history_years": coverage.get("median_history_years")}, "calendar": {"status": calendar_status, "updated_at_utc": calendar_updated, "holiday_count": len(calendar.get("holidays", [])), "special_sessions": calendar.get("special_sessions", [])}, "index": {"status": market_status, "latest_session": index_latest, "reference_ohlcv_session": market.get("reference_ohlcv_session"), "stale_rows": market.get("stale_index_rows", 0), "duplicate_rows": market.get("duplicate_index_rows", 0), "future_rows": market.get("future_index_rows", 0), "risk_level": market.get("latest_record", {}).get("risk_level"), "regime": market.get("latest_record", {}).get("regime"), "breadth_pct_above_sma20": market.get("latest_record", {}).get("breadth_pct_above_sma20")}},
        "reconciliation": {"status": _status(pnl.get("status")), "buy_rows": pnl.get("buy_rows"), "complete_rows": pnl.get("complete_rows"), "incomplete_rows": pnl.get("incomplete_rows"), "mismatch_rows": pnl.get("reconciliation_mismatch_rows"), "last_complete_session": pnl.get("last_complete_session"), "last_incomplete_session": pnl.get("last_incomplete_session")},
        "governance": {"promotion_status": promotion_status, "unmet_gates": promotion.get("unmet_gates", []), "drift_status": governance.get("drift_status", "UNKNOWN"), "model_drift_status": governance.get("model_drift_status", "UNKNOWN"), "safety_status": governance.get("safety_status", "UNKNOWN"), "unstable_stocks": governance.get("unstable_stocks"), "stable_features": governance.get("stable_features"), "performance_findings": performance_findings, "progress_audit_counts": progress_counts, "phase2_status": phase2.get("phase2_status", "UNKNOWN"), "phase2_reason": phase2.get("phase2_reason", "not available")},
    })


def main() -> None:
    payload = build_dashboard()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Dashboard data written: {OUTPUT.relative_to(ROOT)} ({payload['predictions']['count']} latest predictions)")


if __name__ == "__main__":
    main()
