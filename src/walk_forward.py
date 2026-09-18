from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.pipeline import FEATURE_COLUMNS, TARGETS, features, add_targets, rank_stocks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_FILE = DATA / "walk_forward_evaluations.csv"
SUMMARY_FILE = DATA / "walk_forward_summary.csv"

MIN_TRAIN_ROWS = 500
LOOKBACK_MONTHS = 24
CHECKPOINTS = 24


def _fit_models(train_rows: pd.DataFrame) -> dict[str, HistGradientBoostingRegressor]:
    models = {}
    for name, target in TARGETS.items():
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train_rows[FEATURE_COLUMNS], train_rows[target])
        models[name] = model
    return models


def _checkpoints(dates: pd.Series) -> list[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().unique())
    unique = unique.sort_values()
    if len(unique) < 2:
        return []
    start = unique.max() - pd.DateOffset(months=LOOKBACK_MONTHS)
    eligible = unique[unique >= start]
    idx = np.linspace(0, len(eligible) - 2, min(CHECKPOINTS, max(1, len(eligible) - 1)), dtype=int)
    return list(eligible.iloc[np.unique(idx)])


def run_walk_forward() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        raise RuntimeError("ohlcv.csv is missing")

    hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    if hist.empty:
        raise RuntimeError("No historical OHLC data available")

    feat = features(hist)
    work = add_targets(feat).dropna(subset=FEATURE_COLUMNS + list(TARGETS.values())).copy()
    dates = pd.Series(pd.to_datetime(feat["date"], errors="coerce").dropna().dt.normalize().unique())
    checkpoints = _checkpoints(dates)
    rows: list[dict] = []

    for checkpoint in checkpoints:
        train_rows = work[work["date"] < checkpoint]
        if len(train_rows) < MIN_TRAIN_ROWS:
            continue

        session = feat[feat["date"] == checkpoint].dropna(subset=FEATURE_COLUMNS).copy()
        if session.empty:
            continue

        # Rank using only technical information available at the checkpoint.
        ranking = rank_stocks(feat[feat["date"] <= checkpoint], None)
        top = ranking.head(10)[["symbol", "rank", "total_score"]]
        session = session.merge(top, on="symbol", how="inner")
        if len(session) < 10:
            continue

        next_dates = dates[dates > checkpoint]
        if next_dates.empty:
            continue
        target_date = next_dates.min()
        actual = hist[hist["date"] == target_date][["symbol", "open", "high", "low", "close"]]
        session = session.merge(actual, on="symbol", how="inner", suffixes=("", "_actual"))
        if len(session) < 10:
            continue

        # The baseline is the previous close: a simple no-change prediction for every OHLC field.
        session["baseline_open"] = session["close"]
        session["baseline_high"] = session["close"]
        session["baseline_low"] = session["close"]
        session["baseline_close"] = session["close"]

        models = _fit_models(train_rows)
        for name, model in models.items():
            session[f"predicted_{name}"] = session["close"] * (1 + model.predict(session[FEATURE_COLUMNS]))

        session["predicted_high"] = session[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
        session["predicted_low"] = session[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)

        for _, row in session.iterrows():
            result = {
                "prediction_date": checkpoint,
                "target_date": target_date,
                "symbol": row["symbol"],
                "rank": int(row["rank"]),
                "score": float(row["total_score"]),
                "base_close": float(row["close"]),
            }
            for field in ["open", "high", "low", "close"]:
                actual_value = float(row[f"{field}_actual"])
                pred = float(row[f"predicted_{field}"])
                baseline = float(row[f"baseline_{field}"])
                result[f"predicted_{field}"] = pred
                result[f"actual_{field}"] = actual_value
                result[f"{field}_abs_pct_error"] = (
                    abs(actual_value - pred) / abs(actual_value) if actual_value else np.nan
                )
                result[f"baseline_{field}_abs_pct_error"] = (
                    abs(actual_value - baseline) / abs(actual_value) if actual_value else np.nan
                )

            predicted_return = result["predicted_close"] / result["base_close"] - 1
            baseline_return = 0.0
            actual_return = result["actual_close"] / result["base_close"] - 1
            result["close_direction_correct"] = int(np.sign(predicted_return) == np.sign(actual_return))
            result["baseline_close_direction_correct"] = int(
                np.sign(baseline_return) == np.sign(actual_return)
            )
            rows.append(result)

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No walk-forward checkpoints produced usable results")

    output = output.drop_duplicates(["target_date", "symbol"]).sort_values(["target_date", "rank"])
    output.to_csv(OUTPUT_FILE, index=False)

    summary = output.groupby("target_date").agg(
        stocks=("symbol", "count"),
        open_mape=("open_abs_pct_error", "mean"),
        high_mape=("high_abs_pct_error", "mean"),
        low_mape=("low_abs_pct_error", "mean"),
        close_mape=("close_abs_pct_error", "mean"),
        baseline_open_mape=("baseline_open_abs_pct_error", "mean"),
        baseline_high_mape=("baseline_high_abs_pct_error", "mean"),
        baseline_low_mape=("baseline_low_abs_pct_error", "mean"),
        baseline_close_mape=("baseline_close_abs_pct_error", "mean"),
        close_direction_accuracy=("close_direction_correct", "mean"),
        baseline_close_direction_accuracy=("baseline_close_direction_correct", "mean"),
    ).reset_index()

    for field in ["open", "high", "low", "close"]:
        summary[f"{field}_mape_pct"] = summary[f"{field}_mape"] * 100
        summary[f"baseline_{field}_mape_pct"] = summary[f"baseline_{field}_mape"] * 100

    summary["close_direction_accuracy_pct"] = summary["close_direction_accuracy"] * 100
    summary["baseline_close_direction_accuracy_pct"] = (
        summary["baseline_close_direction_accuracy"] * 100
    )
    summary["close_mape_delta_pct"] = (
        summary["close_mape_pct"] - summary["baseline_close_mape_pct"]
    )
    summary["close_direction_delta_pct_points"] = (
        summary["close_direction_accuracy_pct"]
        - summary["baseline_close_direction_accuracy_pct"]
    )
    summary.to_csv(SUMMARY_FILE, index=False)
    return output


if __name__ == "__main__":
    result = run_walk_forward()
    print(
        f"Walk-forward complete: {len(result)} predictions, "
        f"{result['target_date'].nunique()} sessions, "
        f"model close MAPE={result['close_abs_pct_error'].mean() * 100:.2f}%, "
        f"baseline close MAPE={result['baseline_close_abs_pct_error'].mean() * 100:.2f}%, "
        f"model direction accuracy={result['close_direction_correct'].mean() * 100:.2f}%, "
        f"baseline direction accuracy={result['baseline_close_direction_correct'].mean() * 100:.2f}%"
    )
