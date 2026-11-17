"""Chronological out-of-sample comparison of stock models with/without index context.

This is a research evaluation only. It never changes production promotion flags.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT_FILE = DATA / "index_feature_validation.json"
BASELINE_FEATURES = [
    "return_1d", "return_5d", "return_20d", "sma20", "sma50", "ema20", "ema50",
    "rsi14", "volume_ratio", "atr14_pct", "macd", "macd_signal", "bb_position",
    "range_pct", "close_sma20_gap", "close_sma50_gap", "volatility20", "volume_trend5",
]


def _metrics(frame: pd.DataFrame) -> dict:
    x = frame.copy()
    actual = pd.to_numeric(x["actual"], errors="coerce")
    predicted = pd.to_numeric(x["predicted"], errors="coerce")
    valid = actual.notna() & predicted.notna() & np.isfinite(actual) & np.isfinite(predicted)
    x = x.loc[valid].copy()
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    if x.empty:
        return {"rows": 0}
    daily_ic, top_returns, bottom_returns = [], [], []
    for _, day in x.groupby("date", sort=True):
        if len(day) < 5:
            continue
        corr = spearmanr(day["predicted"], day["actual"]).statistic
        if np.isfinite(corr):
            daily_ic.append(float(corr))
        ranked = day.sort_values("predicted", ascending=False)
        n = max(1, int(np.ceil(len(ranked) * 0.10)))
        top_returns.append(float(ranked.head(n)["actual"].mean()))
        bottom_returns.append(float(ranked.tail(n)["actual"].mean()))
    top_mean = float(np.mean(top_returns)) if top_returns else None
    bottom_mean = float(np.mean(bottom_returns)) if bottom_returns else None
    return {
        "rows": int(len(x)),
        "sessions": int(x["date"].nunique()),
        "mae_return_pct": float(np.mean(np.abs(actual - predicted)) * 100.0),
        "rmse_return_pct": float(np.sqrt(np.mean(np.square(actual - predicted))) * 100.0),
        "directional_accuracy_pct": float(np.mean(np.sign(actual) == np.sign(predicted)) * 100.0),
        "mean_daily_rank_ic": float(np.mean(daily_ic)) if daily_ic else None,
        "top_decile_mean_return_pct": top_mean * 100.0 if top_mean is not None else None,
        "bottom_decile_mean_return_pct": bottom_mean * 100.0 if bottom_mean is not None else None,
        "top_minus_bottom_decile_pct": (top_mean - bottom_mean) * 100.0 if top_mean is not None and bottom_mean is not None else None,
    }


def chronological_date_split(frame: pd.DataFrame, train_fraction: float = 0.80) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce").dropna().unique()).sort_values()
    if len(dates) < 30:
        raise ValueError(f"At least 30 unique sessions are required; found {len(dates)}")
    if not 0.60 <= train_fraction <= 0.90:
        raise ValueError("train_fraction must be between 0.60 and 0.90")
    cut = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    return pd.Timestamp(dates[cut - 1]), pd.Timestamp(dates[cut])


def _predict(model: dict, x: pd.DataFrame) -> np.ndarray:
    pred, _ = pipeline._ensemble_predict(model, x)
    return np.asarray(pred, dtype=float)


def run_validation(
    history: pd.DataFrame | None = None,
    index_history: pd.DataFrame | None = None,
    train_fraction: float = 0.80,
    max_train_rows: int = 30000,
    max_test_rows: int = 15000,
) -> dict:
    if history is None:
        history = pd.read_csv(DATA / "ohlcv.csv", parse_dates=["date"])
    if index_history is None:
        index_history = pd.read_csv(DATA / "index_ohlcv.csv", parse_dates=["date"])
    if history.empty or index_history.empty:
        raise RuntimeError("Stock OHLCV and index OHLCV are both required for index-feature validation")
    enriched = pipeline.features(history, index_history=index_history)
    work = pipeline.add_targets(enriched)
    needed = list(dict.fromkeys(BASELINE_FEATURES + pipeline.INDEX_MARKET_FEATURE_COLUMNS + ["target_close_return"]))
    work = work.dropna(subset=needed).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"]).sort_values(["date", "symbol"])
    if len(work) < 1000:
        raise RuntimeError(f"Insufficient aligned stock/index training rows: {len(work)}")
    train_end, test_start = chronological_date_split(work, train_fraction)
    train = work.loc[work["date"] <= train_end]
    test = work.loc[work["date"] >= test_start]
    if len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=42).sort_values(["date", "symbol"])
    if len(test) > max_test_rows:
        test = test.sample(n=max_test_rows, random_state=43).sort_values(["date", "symbol"])
    if len(train) < 500 or len(test) < 100:
        raise RuntimeError(f"Insufficient split size: train={len(train)}, test={len(test)}")
    baseline_model = pipeline._fit_ensemble(train[BASELINE_FEATURES], train["target_close_return"])
    index_model = pipeline._fit_ensemble(train[pipeline.INDEX_AUGMENTED_FEATURE_COLUMNS], train["target_close_return"])
    evaluation = test[["date", "symbol", "target_close_return"]].rename(columns={"target_close_return": "actual"}).copy()
    evaluation["baseline_predicted"] = _predict(baseline_model, test[BASELINE_FEATURES])
    evaluation["index_predicted"] = _predict(index_model, test[pipeline.INDEX_AUGMENTED_FEATURE_COLUMNS])
    evaluation["period"] = pd.qcut(
        evaluation["date"].rank(method="dense"), q=3, labels=["test_1", "test_2", "test_3"],
        duplicates="drop",
    ).astype(str)
    results = {}
    for period, segment in evaluation.groupby("period", sort=True):
        results[period] = {
            "baseline": _metrics(segment.rename(columns={"baseline_predicted": "predicted"})[["date", "actual", "predicted"]]),
            "with_index_features": _metrics(segment.rename(columns={"index_predicted": "predicted"})[["date", "actual", "predicted"]]),
        }
    overall = {
        "baseline": _metrics(evaluation.rename(columns={"baseline_predicted": "predicted"})[["date", "actual", "predicted"]]),
        "with_index_features": _metrics(evaluation.rename(columns={"index_predicted": "predicted"})[["date", "actual", "predicted"]]),
    }
    base_mae = overall["baseline"].get("mae_return_pct")
    index_mae = overall["with_index_features"].get("mae_return_pct")
    mae_improved = index_mae is not None and base_mae is not None and index_mae < base_mae * 0.99
    base_ic = overall["baseline"].get("mean_daily_rank_ic")
    index_ic = overall["with_index_features"].get("mean_daily_rank_ic")
    rank_not_worse = index_ic is not None and base_ic is not None and index_ic >= base_ic
    improved_windows = 0
    for item in results.values():
        b, a = item["baseline"], item["with_index_features"]
        if (a.get("mae_return_pct") is not None and b.get("mae_return_pct") is not None
                and a["mae_return_pct"] < b["mae_return_pct"]
                and a.get("mean_daily_rank_ic") is not None
                and b.get("mean_daily_rank_ic") is not None
                and a["mean_daily_rank_ic"] >= b["mean_daily_rank_ic"]):
            improved_windows += 1
    report = {
        "status": "EVALUATED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "chronological 80/20 date split; fixed 3 expanding-order test windows; same ensemble and same rows for baseline and augmented model",
        "target": "next observed stock-session close return",
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sessions": int(train["date"].nunique()),
        "test_sessions": int(test["date"].nunique()),
        "baseline_feature_count": len(BASELINE_FEATURES),
        "index_augmented_feature_count": len(pipeline.INDEX_AUGMENTED_FEATURE_COLUMNS),
        "index_feature_columns": pipeline.INDEX_MARKET_FEATURE_COLUMNS,
        "index_feature_availability_pct": {
            index: float(work[f"index_{index.lower()}_available_1d"].mean() * 100.0)
            for index in pipeline.INDEX_FEATURE_INDEXES
        },
        "overall": overall,
        "chronological_test_windows": results,
        "windows_improved_on_both_mae_and_rank_ic": improved_windows,
        "decision": "PROMISING_REQUIRES_FURTHER_VALIDATION" if mae_improved and rank_not_worse and improved_windows == len(results) else "NO_CLEAR_IMPROVEMENT",
        "production_promotion_authorized": False,
        "constituent_weight_features_used": False,
        "constituent_note": "Dated constituent relationships remain a separate research layer; only same-session historical index return features are used here. Historical weights/membership are not backfilled.",
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_validation()
