from __future__ import annotations

"""Phases 41-43: V2 selection diagnosis and offline candidate experiments.

All outputs are diagnostic only. These routines do not change the V1 champion,
production filters, ranking, position sizing, or V2 eligibility.
Phase 43 is a fixed-grid sensitivity report over recorded predictions, not a
model-refit walk-forward backtest and not permission to deploy thresholds.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def audit_selection_funnel(funnel: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Describe observed stage counts without fabricating missing gate counts."""
    if funnel.empty or not {"stage", "input_rows", "output_rows"}.issubset(funnel.columns):
        if trades.empty:
            rows = [{"stage": "selection_funnel", "input_rows": 0, "output_rows": 0,
                     "rejected_rows": 0, "detail": "Awaiting a V2 engine run; internal gate counts are not yet observed"}]
            return pd.DataFrame(rows), {"status": "AWAITING_NEXT_V2_RUN", "observed_stages": 0}
        signal = trades.get("signal", pd.Series("", index=trades.index)).astype(str).str.upper()
        return pd.DataFrame([{
            "stage": "history_only_fallback", "input_rows": int(len(trades)),
            "output_rows": int(signal.eq("BUY").sum()), "rejected_rows": int(signal.eq("SKIP").sum()),
            "detail": "History-only fallback; it cannot attribute rejections to individual gates",
        }]), {"status": "FALLBACK_HISTORY_ONLY", "observed_stages": 0,
               "buy_rows": int(signal.eq("BUY").sum()), "skip_rows": int(signal.eq("SKIP").sum())}

    out = funnel.copy()
    for col in ("input_rows", "output_rows", "rejected_rows"):
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["count_consistent"] = (
        out["input_rows"].notna() & out["output_rows"].notna()
        & (out["output_rows"] <= out["input_rows"])
        & (out["rejected_rows"].isna() | (out["rejected_rows"] == out["input_rows"] - out["output_rows"]))
    )
    bad = out.loc[~out["count_consistent"], "stage"].astype(str).tolist()
    summary = {
        "status": "OBSERVED" if not bad else "REVIEW_COUNT_INCONSISTENCIES",
        "observed_stages": int(len(out)),
        "stage_counts_consistent": not bool(bad),
        "inconsistent_stages": bad,
        "final_selected_rows": int(out.iloc[-1]["output_rows"]) if pd.notna(out.iloc[-1]["output_rows"]) else None,
        "data_source": "paper-trading V2 execution telemetry",
    }
    return out, summary


def compare_selected_rejected(candidates: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare Phase-2 selected and rejected saved candidates against realized close outcomes."""
    required = {"target_date", "symbol", "base_close", "predicted_close", "phase2_selected"}
    actual_required = {"date", "symbol", "close"}
    if candidates.empty or actuals.empty or not required.issubset(candidates.columns) or not actual_required.issubset(actuals.columns):
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "matched_rows": 0}

    p = candidates.copy()
    a = actuals[["date", "symbol", "close"]].copy()
    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    a["symbol"] = a["symbol"].astype(str).str.upper().str.strip()
    for c in ("base_close", "predicted_close", "phase2_selected"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    a["close"] = pd.to_numeric(a["close"], errors="coerce")
    p = p.dropna(subset=["target_date", "symbol", "base_close", "predicted_close", "phase2_selected"])
    a = a.dropna(subset=["date", "symbol", "close"]).drop_duplicates(["date", "symbol"], keep="last")
    a = a.rename(columns={"date": "target_date", "close": "actual_close"})
    x = p.merge(a, on=["target_date", "symbol"], how="inner", validate="many_to_one")
    if x.empty:
        return x, {"status": "INSUFFICIENT_DATA", "matched_rows": 0}
    x["group"] = np.where(x["phase2_selected"].eq(1), "SELECTED", "REJECTED")
    x["realized_close_return_pct"] = (x["actual_close"] / x["base_close"].replace(0, np.nan) - 1) * 100
    x["predicted_close_return_pct"] = (x["predicted_close"] / x["base_close"].replace(0, np.nan) - 1) * 100
    x["direction_correct"] = (
        (x["predicted_close"] >= x["base_close"]) == (x["actual_close"] >= x["base_close"])
    ).astype(int)
    report = x.groupby("group", as_index=False).agg(
        candidates=("symbol", "size"),
        target_dates=("target_date", "nunique"),
        mean_realized_close_return_pct=("realized_close_return_pct", "mean"),
        median_realized_close_return_pct=("realized_close_return_pct", "median"),
        realized_positive_rate_pct=("realized_close_return_pct", lambda s: float((s > 0).mean() * 100)),
        prediction_direction_accuracy_pct=("direction_correct", lambda s: float(s.mean() * 100)),
        mean_abs_prediction_return_pct=("predicted_close_return_pct", lambda s: float(s.abs().mean())),
    )
    summary = {
        "status": "DESCRIPTIVE_ONLY",
        "matched_rows": int(len(x)),
        "selected_rows": int(x["group"].eq("SELECTED").sum()),
        "rejected_rows": int(x["group"].eq("REJECTED").sum()),
        "method": "realized target-date close versus saved base close; descriptive association, not causal selector lift",
        "production_settings_changed": False,
    }
    return report, summary


def fixed_grid_threshold_sensitivity(candidates: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a predeclared confidence/ATR sensitivity grid on chronological holdout rows."""
    required = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "confidence_v3"}
    actual_required = {"date", "symbol", "open", "close"}
    if candidates.empty or actuals.empty or not required.issubset(candidates.columns) or not actual_required.issubset(actuals.columns):
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "experiments": 0,
                                "method": "fixed-grid sensitivity; no model refit"}
    p = candidates.copy()
    a = actuals[["date", "symbol", "open", "close"]].copy()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce").dt.normalize()
    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
    for frame in (p, a):
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    for c in ("base_close", "predicted_close", "confidence_v3"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    for c in ("open", "close"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    p = p.dropna(subset=["prediction_date", "target_date", "symbol", "base_close", "predicted_close", "confidence_v3"])
    p = p[p["prediction_date"] < p["target_date"]].drop_duplicates(["target_date", "symbol"], keep="last")
    a = a.dropna(subset=["date", "symbol", "open", "close"]).drop_duplicates(["date", "symbol"], keep="last")
    a = a.rename(columns={"date": "target_date", "open": "actual_open", "close": "actual_close"})
    x = p.merge(a, on=["target_date", "symbol"], how="inner", validate="many_to_one")
    if x.empty:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "experiments": 0,
                                "method": "fixed-grid sensitivity; no model refit"}
    x["confidence_pct"] = x.groupby("target_date")["confidence_v3"].rank(pct=True, method="average")
    x["realized_return_pct"] = (x["actual_close"] / x["actual_open"].replace(0, np.nan) - 1) * 100
    x["predicted_direction"] = np.sign(x["predicted_close"] - x["base_close"])
    x["actual_direction"] = np.sign(x["actual_close"] - x["base_close"])
    x["direction_correct"] = (x["predicted_direction"] == x["actual_direction"]).astype(float)
    dates = sorted(x["target_date"].dropna().unique())
    if len(dates) < 3:
        return pd.DataFrame(), {"status": "INSUFFICIENT_CHRONOLOGY", "experiments": 0,
                                "target_dates": len(dates), "method": "fixed-grid sensitivity; no model refit"}
    split = min(len(dates) - 1, max(1, int(np.ceil(len(dates) * 0.70))))
    holdout_start = pd.Timestamp(dates[split])
    x["period"] = np.where(x["target_date"] < holdout_start, "EARLY_CONTEXT", "CHRONOLOGICAL_HOLDOUT")
    rows = []
    for confidence_floor in (0.40, 0.50, 0.60, 0.70):
        for atr_cap in ("NO_ATR_CAP",):
            # ATR is deliberately not inferred from columns that are absent.
            # This first grid tests confidence selection only; a future ATR grid
            # requires a validated, prediction-date-aligned ATR field.
            for period in ("EARLY_CONTEXT", "CHRONOLOGICAL_HOLDOUT"):
                part = x[x["period"].eq(period)]
                selected = part[part["confidence_pct"] >= confidence_floor]
                rows.append({
                    "confidence_percentile_floor": confidence_floor,
                    "atr_cap": atr_cap,
                    "period": period,
                    "selected_rows": int(len(selected)),
                    "target_dates": int(selected["target_date"].nunique()),
                    "mean_realized_intraday_return_pct": float(selected["realized_return_pct"].mean()) if not selected.empty else None,
                    "median_realized_intraday_return_pct": float(selected["realized_return_pct"].median()) if not selected.empty else None,
                    "positive_return_rate_pct": float((selected["realized_return_pct"] > 0).mean() * 100) if not selected.empty else None,
                    "direction_accuracy_pct": float(selected["direction_correct"].mean() * 100) if not selected.empty else None,
                    "status": "DESCRIPTIVE_SMALL_SAMPLE" if len(selected) < 20 or selected["target_date"].nunique() < 5 else "DESCRIPTIVE",
                })
    report = pd.DataFrame(rows)
    holdout_rows = int(x["period"].eq("CHRONOLOGICAL_HOLDOUT").sum())
    summary = {
        "status": "SENSITIVITY_ONLY" if holdout_rows >= 20 and x.loc[x["period"].eq("CHRONOLOGICAL_HOLDOUT"), "target_date"].nunique() >= 5 else "INSUFFICIENT_HOLDOUT",
        "experiments": int(len(report)),
        "holdout_rows": holdout_rows,
        "holdout_target_dates": int(x.loc[x["period"].eq("CHRONOLOGICAL_HOLDOUT"), "target_date"].nunique()),
        "holdout_start_date": holdout_start.date().isoformat(),
        "method": "fixed confidence percentile grid on saved predictions; chronological holdout; no model refit",
        "production_settings_changed": False,
        "thresholds_deployed": False,
        "promotion_allowed": False,
        "atr_threshold_experiment": "NOT_RUN: no validated prediction-date-aligned ATR field required by this module",
    }
    return report, summary


def run() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    funnel, funnel_summary = audit_selection_funnel(_read("phase32_v2_selection_funnel.csv"), _read("paper_trades_v2.csv"))
    funnel.to_csv(DATA / "phase41_v2_funnel_audit.csv", index=False)

    selected, selected_summary = compare_selected_rejected(
        _read("phase2_optimized_candidates.csv"), _read("ohlcv.csv")
    )
    selected.to_csv(DATA / "phase42_selected_vs_rejected.csv", index=False)

    sensitivity, sensitivity_summary = fixed_grid_threshold_sensitivity(
        _read("phase2_optimized_candidates.csv"), _read("ohlcv.csv")
    )
    sensitivity.to_csv(DATA / "phase43_threshold_sensitivity.csv", index=False)

    summary = {
        "phases": [41, 42, 43],
        "phase41_funnel": funnel_summary,
        "phase42_candidate_quality": selected_summary,
        "phase43_threshold_sensitivity": sensitivity_summary,
        "production_champion": "V1",
        "v2_promoted": False,
        "production_settings_changed": False,
        "ranking_or_risk_changed": False,
        "automatic_threshold_deployment": False,
    }
    (DATA / "phase41_43_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
