from __future__ import annotations

"""Analysis-only V2, direction-quality and storage diagnostics."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _read(name):
    path = DATA / name
    try:
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def v2_diagnostic():
    x = _read("paper_trades_v2.csv")
    if x.empty or "signal" not in x:
        return {"status": "INSUFFICIENT_DATA", "rows": len(x), "buy_rows": 0, "production_filters_changed": False}, pd.DataFrame()
    sig = x["signal"].fillna("").astype(str).str.upper().str.strip()
    skips = x.loc[sig.eq("SKIP")].copy()
    reasons = skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip().replace("", "MISSING_REASON")
    phase4 = skips.get("phase4_reject_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip().replace("", "NOT_RECORDED")
    skips["primary_reason"] = reasons
    skips["phase4_reason"] = phase4
    detail = skips.groupby(["primary_reason", "phase4_reason"], dropna=False).size().reset_index(name="rows")
    funnel = _read("phase32_v2_selection_funnel.csv")
    if not funnel.empty and {"stage", "input_rows", "output_rows"}.issubset(funnel.columns):
        extra = pd.DataFrame([{"primary_reason": "FUNNEL_STAGE", "phase4_reason": str(r.get("stage", "UNKNOWN")), "rows": max(0, int(pd.to_numeric(r["input_rows"], errors="coerce") - pd.to_numeric(r["output_rows"], errors="coerce")))} for _, r in funnel.iterrows()])
        detail = pd.concat([detail, extra], ignore_index=True)
    buys = int(sig.eq("BUY").sum())
    return {"status": "ZERO_BUY_DIAGNOSED" if buys == 0 else "BUY_ACTIVITY_PRESENT", "rows": len(x), "buy_rows": buys, "skip_rows": int(sig.eq("SKIP").sum()), "skip_reason_counts": {str(k): int(v) for k, v in reasons.value_counts().items()}, "missing_skip_reasons": int(reasons.eq("MISSING_REASON").sum()), "production_filters_changed": False, "thresholds_deployed": False}, detail


def direction_diagnostic():
    x = _read("phase33_chronological_prediction_scores.csv")
    if x.empty or not {"target_date", "close_direction_correct"}.issubset(x.columns):
        return {"status": "INSUFFICIENT_DATA", "scored_rows": 0, "directional_accuracy_pct": None, "recent_5_directional_accuracy_pct": None, "production_model_changed": False}, pd.DataFrame()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["close_direction_correct"] = pd.to_numeric(x["close_direction_correct"], errors="coerce")
    x = x.dropna(subset=["target_date", "close_direction_correct"])
    by_date = x.groupby("target_date", as_index=False).agg(scored_rows=("close_direction_correct", "size"), direction_accuracy_pct=("close_direction_correct", lambda s: float(s.mean() * 100))).sort_values("target_date")
    recent = x[x["target_date"].isin(by_date["target_date"].tail(5))]
    overall = float(x["close_direction_correct"].mean() * 100) if len(x) else None
    recent_acc = float(recent["close_direction_correct"].mean() * 100) if len(recent) else None
    return {"status": "REVIEW_DIRECTION_QUALITY" if recent_acc is not None and recent_acc < 50 else "DESCRIPTIVE_ONLY", "scored_rows": len(x), "target_dates": int(x["target_date"].nunique()), "directional_accuracy_pct": overall, "recent_5_directional_accuracy_pct": recent_acc, "production_model_changed": False, "method": "saved chronological scores; no refit or threshold search"}, by_date


def storage_audit():
    p = DATA / "ohlcv.csv"
    if not p.exists():
        return {"status": "FILE_MISSING", "storage_changed": False}
    size = p.stat().st_size
    try:
        x = pd.read_csv(p, usecols=lambda c: c in {"date", "symbol"}, low_memory=False)
        duplicates = int((x["symbol"].astype(str).str.upper().str.strip() + "|" + x["date"].astype(str).str.strip()).duplicated().sum()) if {"symbol", "date"}.issubset(x.columns) else None
        dates = pd.to_datetime(x["date"], errors="coerce").dropna() if "date" in x.columns else pd.Series(dtype="datetime64[ns]")
        span = {"first": dates.min().date().isoformat(), "last": dates.max().date().isoformat()} if len(dates) else None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {"status": "AUDIT_READ_FAILED", "bytes": size, "error_type": type(exc).__name__, "storage_changed": False}
    over = size > 50 * 1024 * 1024
    return {"status": "ABOVE_GITHUB_RECOMMENDED_SIZE" if over else "WITHIN_GITHUB_RECOMMENDED_SIZE", "path": "data/ohlcv.csv", "bytes": size, "size_mb": round(size / 1048576, 2), "rows": len(x), "duplicate_symbol_date_keys": duplicates, "date_span": span, "recommended_action": "Plan and test a staged compressed/partitioned storage migration with rollback; do not replace the source file in this diagnostic." if over else "No size-driven migration indicated.", "storage_changed": False, "source_file_modified": False}


def run():
    DATA.mkdir(exist_ok=True)
    v2, reasons = v2_diagnostic()
    direction, dates = direction_diagnostic()
    storage = storage_audit()
    if not reasons.empty:
        reasons.to_csv(DATA / "phase73_v2_zero_buy_breakdown.csv", index=False)
    if not dates.empty:
        dates.assign(target_date=dates["target_date"].dt.strftime("%Y-%m-%d")).to_csv(DATA / "phase74_direction_by_date.csv", index=False)
    report = {"as_of_utc": pd.Timestamp.now(tz="UTC").isoformat(), "phases": [73, 74, 75, 76], "phase73_v2_zero_buy": v2, "phase74_directional_quality": direction, "phase75_evidence": {"status": "RECORDED" if direction.get("scored_rows", 0) else "INSUFFICIENT_DATA", "chronological_scoring": True, "new_model_fit": False}, "phase76_storage_audit": storage, "safety": {"production_champion": "V1", "v2_mode": "SHADOW_ONLY", "automatic_promotion": False, "ranking_or_risk_changed": False, "storage_migrated": False}}
    (DATA / "phase73_76_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    pd.DataFrame([{ "phase": k, **{a:b for a,b in v.items() if not isinstance(b,(dict,list))}} for k,v in [("73",v2),("74",direction),("76",storage)]]).to_csv(DATA / "phase73_76_diagnostics.csv", index=False)
    print(json.dumps(report, indent=2, allow_nan=False, default=str))
    return report

if __name__ == "__main__":
    run()
