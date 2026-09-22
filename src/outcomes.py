from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_FILE = DATA / "prediction_outcomes.csv"

HORIZONS = {
    "3m": pd.DateOffset(months=3),
    "6m": pd.DateOffset(months=6),
    "9m": pd.DateOffset(months=9),
    "12m": pd.DateOffset(months=12),
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"], errors="coerce").dt.normalize()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["symbol"] = history["symbol"].astype(str).str.strip()
    return predictions, history


def _future_close(history: pd.DataFrame, symbol: str, target: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    rows = history[(history["symbol"] == symbol) & (history["date"] >= target)].sort_values("date")
    if rows.empty:
        return pd.NaT, np.nan
    row = rows.iloc[0]
    return pd.Timestamp(row["date"]).normalize(), float(row["close"])


def run_outcomes() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or not HISTORY_FILE.exists():
        print("Outcome tracking skipped: predictions or history file missing.")
        return pd.DataFrame()

    predictions, history = _load()
    if predictions.empty:
        print("Outcome tracking skipped: no predictions.")
        return pd.DataFrame()

    rows: list[dict] = []
    for _, p in predictions.iterrows():
        target_date = pd.Timestamp(p["target_date"]) if pd.notna(p["target_date"]) else pd.NaT
        base_close = pd.to_numeric(pd.Series([p.get("base_close")]), errors="coerce").iloc[0]
        if pd.isna(target_date) or pd.isna(base_close) or base_close == 0:
            continue

        row = {
            "prediction_date": p["prediction_date"],
            "target_date": target_date,
            "symbol": str(p["symbol"]).strip(),
            "rank": p.get("rank"),
            "score": p.get("score"),
            "base_close": float(base_close),
            "predicted_close": p.get("predicted_close"),
        }
        for label, offset in HORIZONS.items():
            requested = target_date + offset
            actual_date, actual_close = _future_close(history, row["symbol"], requested)
            row[f"{label}_target_date"] = actual_date
            row[f"{label}_actual_close"] = actual_close
            row[f"{label}_return_pct"] = (
                (actual_close / base_close - 1.0) * 100.0
                if np.isfinite(actual_close)
                else np.nan
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    if OUTPUT_FILE.exists():
        old = pd.read_csv(OUTPUT_FILE)
        output = pd.concat([old, output], ignore_index=True)

    output["prediction_date"] = pd.to_datetime(output["prediction_date"], errors="coerce").dt.normalize()
    output["target_date"] = pd.to_datetime(output["target_date"], errors="coerce").dt.normalize()
    for label in HORIZONS:
        output[f"{label}_target_date"] = pd.to_datetime(output[f"{label}_target_date"], errors="coerce").dt.normalize()

    output = (
        output.dropna(subset=["target_date", "symbol"])
        .drop_duplicates(["target_date", "symbol"], keep="last")
        .sort_values(["target_date", "rank", "symbol"])
        .reset_index(drop=True)
    )
    output.to_csv(OUTPUT_FILE, index=False)
    completed = sum(int(output[f"{label}_actual_close"].notna().sum()) for label in HORIZONS)
    print(f"Prediction outcomes updated: {len(output)} predictions, {completed} horizon observations available.")
    return output


if __name__ == "__main__":
    run_outcomes()
