from __future__ import annotations

"""Analysis-only forecast quality audit; never changes production predictions or gates."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCORE_FILE = "phase33_chronological_prediction_scores.csv"
OHLCV_FILE = "ohlcv.csv"


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna() & a.ne(0)
    return float((p[valid] - a[valid]).abs().div(a[valid].abs()).mean() * 100) if valid.any() else None


def _mae(actual: pd.Series, predicted: pd.Series) -> float | None:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    return float((p[valid] - a[valid]).abs().mean()) if valid.any() else None


def _direction_accuracy(frame: pd.DataFrame, predicted_col: str, base_col: str = "base_close") -> float | None:
    required = {"actual_close", predicted_col, base_col}
    if not required.issubset(frame.columns):
        return None
    # A previous-close forecast is explicitly neutral (no predicted move).
    # It has no directional call to score; comparing it to actual direction
    # incorrectly reported a near-zero "accuracy" due to floating-point noise.
    if predicted_col == base_col:
        return None
    actual = pd.to_numeric(frame["actual_close"], errors="coerce")
    predicted = pd.to_numeric(frame[predicted_col], errors="coerce")
    base = pd.to_numeric(frame[base_col], errors="coerce")
    valid = actual.notna() & predicted.notna() & base.notna()
    if not valid.any():
        return None
    return float(((predicted[valid] - base[valid]).apply(np.sign) == (actual[valid] - base[valid]).apply(np.sign)).mean() * 100)


def _historical_momentum_baseline(scores: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """One-step baseline using only rows dated on/before prediction_date."""
    result = pd.Series(np.nan, index=scores.index, dtype=float)
    required = {"date", "symbol", "close"}
    if history.empty or not required.issubset(history.columns):
        return result
    h = history.loc[:, ["date", "symbol", "close"]].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.strip().str.upper()
    h["close"] = pd.to_numeric(h["close"], errors="coerce")
    h = h.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])
    h = h.drop_duplicates(["symbol", "date"], keep="last")
    by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in h.groupby("symbol", sort=False)}
    for idx, row in scores.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        pred_date = pd.to_datetime(row.get("prediction_date"), errors="coerce")
        base = pd.to_numeric(pd.Series([row.get("base_close")]), errors="coerce").iloc[0]
        if pd.isna(pred_date) or pd.isna(base) or base <= 0 or symbol not in by_symbol:
            continue
        group = by_symbol[symbol]
        prior = group.loc[group["date"].le(pred_date.normalize())]
        # Five completed close-to-close intervals; all input dates precede the target date.
        if len(prior) < 6:
            continue
        last_close = float(prior.iloc[-1]["close"])
        old_close = float(prior.iloc[-6]["close"])
        if old_close > 0:
            result.loc[idx] = float(base * (last_close / old_close))
    return result


def _metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    actual = pd.to_numeric(frame.get("actual_close", pd.Series(np.nan, index=frame.index)), errors="coerce")
    pred = pd.to_numeric(frame.get(pred_col, pd.Series(np.nan, index=frame.index)), errors="coerce")
    valid = actual.notna() & pred.notna() & actual.ne(0)
    direction = None if pred_col in {"base_close", "previous_close_baseline"} else _direction_accuracy(frame, pred_col)
    return {
        "scored_rows": int(valid.sum()),
        "close_mape_pct": _mape(actual, pred),
        "close_mae": _mae(actual, pred),
        "directional_accuracy_pct": direction,
    }


def build_quality_audit(scores: pd.DataFrame, history: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        return ({
            "status": "INSUFFICIENT_DATA", "missing_columns": missing, "scored_rows": 0,
            "production_model_changed": False, "ranking_or_risk_changed": False,
            "v2_mode": "SHADOW_ONLY", "production_champion": "V1",
        }, pd.DataFrame(), pd.DataFrame())

    x = scores.copy().reset_index(drop=True)
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    for col in ("base_close", "predicted_close", "actual_close", "actual_open", "actual_high", "actual_low"):
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"])
    # Exclude impossible temporal rows rather than allowing look-ahead data into the audit.
    temporal_bad = x["target_date"].le(x["prediction_date"])
    rejected_temporal_rows = int(temporal_bad.sum())
    x = x.loc[~temporal_bad].copy()
    x["previous_close_baseline"] = x["base_close"]
    x["five_session_momentum_baseline"] = _historical_momentum_baseline(x, history)
    x["model_ape_pct"] = (x["predicted_close"] - x["actual_close"]).abs().div(x["actual_close"].abs().replace(0, np.nan)) * 100
    x["previous_close_ape_pct"] = (x["previous_close_baseline"] - x["actual_close"]).abs().div(x["actual_close"].abs().replace(0, np.nan)) * 100
    x["momentum_ape_pct"] = (x["five_session_momentum_baseline"] - x["actual_close"]).abs().div(x["actual_close"].abs().replace(0, np.nan)) * 100
    x["model_minus_previous_close_ape_pp"] = x["model_ape_pct"] - x["previous_close_ape_pct"]
    x["model_minus_momentum_ape_pp"] = x["model_ape_pct"] - x["momentum_ape_pct"]

    overall = {
        "status": "SCORED" if len(x) >= 20 else ("SMALL_SAMPLE" if len(x) else "INSUFFICIENT_DATA"),
        "scored_rows": int(len(x)), "symbols": int(x["symbol"].nunique()),
        "target_dates": int(x["target_date"].nunique()),
        "rejected_temporal_rows": rejected_temporal_rows,
        "model": _metrics(x, "predicted_close"),
        "previous_close_baseline": _metrics(x, "previous_close_baseline"),
        "five_session_momentum_baseline": _metrics(x, "five_session_momentum_baseline"),
        "model_beats_previous_close_baseline": bool(_mape(x["actual_close"], x["predicted_close"]) < _mape(x["actual_close"], x["previous_close_baseline"])) if x["actual_close"].notna().any() else False,
        "model_beats_five_session_momentum_baseline": bool(_mape(x["actual_close"], x["predicted_close"]) < _mape(x["actual_close"], x["five_session_momentum_baseline"])) if x["five_session_momentum_baseline"].notna().any() else False,
        "momentum_baseline_scored_rows": int(x["five_session_momentum_baseline"].notna().sum()),
        "method": "Saved chronological predictions; historical momentum uses only closes dated on or before prediction_date. This is an evaluation baseline, not a deployed strategy.",
        "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False, "ranking_or_risk_changed": False,
        "thresholds_changed": False, "automatic_promotion": False,
    }

    symbol_rows = []
    for symbol, group in x.groupby("symbol", sort=True):
        model_mape = _mape(group["actual_close"], group["predicted_close"])
        prev_mape = _mape(group["actual_close"], group["previous_close_baseline"])
        momentum_mape = _mape(group["actual_close"], group["five_session_momentum_baseline"])
        symbol_rows.append({
            "symbol": symbol, "scored_rows": int(len(group)), "target_dates": int(group["target_date"].nunique()),
            "model_close_mape_pct": model_mape, "previous_close_mape_pct": prev_mape,
            "five_session_momentum_mape_pct": momentum_mape,
            "model_minus_previous_close_ape_pp": model_mape - prev_mape if model_mape is not None and prev_mape is not None else None,
            "model_minus_momentum_ape_pp": model_mape - momentum_mape if model_mape is not None and momentum_mape is not None else None,
            "directional_accuracy_pct": _direction_accuracy(group, "predicted_close"),
        })
    by_symbol = pd.DataFrame(symbol_rows)
    if not by_symbol.empty:
        by_symbol = by_symbol.sort_values(["model_minus_previous_close_ape_pp", "scored_rows"], ascending=[False, False], na_position="last")

    segment_rows = []
    for (date, period), group in x.groupby(["target_date", x.get("period", pd.Series("UNKNOWN", index=x.index)).fillna("UNKNOWN").astype(str)], dropna=False, sort=True):
        segment_rows.append({
            "target_date": str(date.date()) if pd.notna(date) else "",
            "period": str(period), "scored_rows": int(len(group)),
            "model_close_mape_pct": _mape(group["actual_close"], group["predicted_close"]),
            "previous_close_mape_pct": _mape(group["actual_close"], group["previous_close_baseline"]),
            "momentum_close_mape_pct": _mape(group["actual_close"], group["five_session_momentum_baseline"]),
            "directional_accuracy_pct": _direction_accuracy(group, "predicted_close"),
            "model_beats_previous_close": _mape(group["actual_close"], group["predicted_close"]) < _mape(group["actual_close"], group["previous_close_baseline"]),
        })
    by_segment = pd.DataFrame(segment_rows)
    return overall, by_symbol, by_segment


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    scores = _read(SCORE_FILE)
    history = _read(OHLCV_FILE)
    summary, by_symbol, by_segment = build_quality_audit(scores, history)
    by_symbol.to_csv(DATA / "phase97_forecast_quality_by_symbol.csv", index=False)
    by_segment.to_csv(DATA / "phase97_forecast_quality_by_segment.csv", index=False)
    (DATA / "phase97_forecast_quality_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
