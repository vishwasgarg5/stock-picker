from __future__ import annotations

"""Phase 118: forecast error diagnosis by symbol and realized-move regime.

Analysis-only diagnostics against the previous-close baseline. Does not alter
production forecasts, rankings, risk controls, or promotion decisions.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORES = DATA / "phase33_chronological_prediction_scores.csv"
SYMBOL_OUT = DATA / "phase118_forecast_errors_by_symbol.csv"
REGIME_OUT = DATA / "phase118_forecast_errors_by_regime.csv"
SUMMARY_OUT = DATA / "phase118_forecast_error_diagnosis.json"
REQUIRED = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "model_mape_pct": None, "baseline_mape_pct": None,
                "delta_mape_pct_points": None, "signed_return_bias_pct": None,
                "model_beats_baseline": False}
    actual = frame["actual_close"]
    base = frame["base_close"]
    pred = frame["predicted_close"]
    model = (pred - actual).abs().div(actual).mean() * 100
    baseline = (base - actual).abs().div(actual).mean() * 100
    bias = ((pred - actual).div(base)).mean() * 100
    return {"rows": int(len(frame)), "model_mape_pct": round(float(model), 6),
            "baseline_mape_pct": round(float(baseline), 6),
            "delta_mape_pct_points": round(float(model - baseline), 6),
            "signed_return_bias_pct": round(float(bias), 6),
            "model_beats_baseline": bool(model < baseline)}


def diagnose_forecast_errors(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing = sorted(REQUIRED.difference(scores.columns))
    if missing:
        empty = pd.DataFrame()
        return empty, empty, {"status": "INSUFFICIENT_DATA", "missing_columns": missing,
            "scored_rows": 0, "production_model_changed": False, "automatic_promotion": False}
    x = scores.copy()
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce")
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce")
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    for col in ("base_close", "predicted_close", "actual_close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=list(REQUIRED))
    x = x[(x["target_date"] > x["prediction_date"]) & x["base_close"].gt(0) & x["actual_close"].gt(0)]
    x = x.drop_duplicates(["prediction_date", "target_date", "symbol"], keep="last")
    x["realized_abs_move_pct"] = (x["actual_close"] / x["base_close"] - 1).abs() * 100
    symbol_rows = []
    for symbol, part in x.groupby("symbol"):
        if len(part) < 3:
            continue
        symbol_rows.append({"symbol": symbol, **_metrics(part)})
    by_symbol = pd.DataFrame(symbol_rows)
    if not by_symbol.empty:
        by_symbol = by_symbol.sort_values(["delta_mape_pct_points", "rows"], ascending=[False, False]).reset_index(drop=True)
    x["move_regime"] = pd.cut(x["realized_abs_move_pct"], bins=[-np.inf, 1.0, 2.5, np.inf],
                               labels=["QUIET_LT_1PCT", "NORMAL_1_TO_2_5PCT", "VOLATILE_GE_2_5PCT"])
    regime_rows = []
    for regime, part in x.groupby("move_regime", observed=True):
        regime_rows.append({"regime": str(regime), **_metrics(part)})
    by_regime = pd.DataFrame(regime_rows)
    period_rows = []
    if "period" in x:
        for period, part in x.groupby("period", dropna=False):
            period_rows.append({"period": str(period), **_metrics(part)})
    summary = {"status": "SCORED" if len(x) >= 20 else "SMALL_SAMPLE",
        "scored_rows": int(len(x)), "symbols": int(x["symbol"].nunique()),
        "target_sessions": int(x["target_date"].nunique()),
        "symbol_rows_reported_minimum_3_observations": int(len(by_symbol)),
        "regime_definition": "Absolute realized close move from base_close: quiet <1%, normal 1%-<2.5%, volatile >=2.5%.",
        "overall": _metrics(x), "by_period": period_rows,
        "worst_symbols": by_symbol.head(10).to_dict(orient="records") if not by_symbol.empty else [],
        "best_symbols": by_symbol.sort_values("delta_mape_pct_points").head(10).to_dict(orient="records") if not by_symbol.empty else [],
        "regimes": by_regime.to_dict(orient="records") if not by_regime.empty else [],
        "production_model_changed": False, "ranking_or_risk_changed": False,
        "automatic_promotion": False, "interpretation": "Diagnostic only. Symbol and realized-move segments are descriptive, not causal; small per-symbol samples must not be used alone to exclude stocks."}
    return by_symbol, by_regime, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        scores = pd.read_csv(SCORES, low_memory=False) if SCORES.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        scores = pd.DataFrame()
    by_symbol, by_regime, summary = diagnose_forecast_errors(scores)
    by_symbol.to_csv(SYMBOL_OUT, index=False)
    by_regime.to_csv(REGIME_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
