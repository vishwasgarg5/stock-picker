from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES_FILE = DATA / "prediction_candidates_history.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_FILE = DATA / "selection_validation.csv"

MIN_ROWS = 100
MIN_SESSIONS = 12
MIN_IMPROVEMENT = 0.01  # 1% relative MAPE improvement


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_selection_validation() -> pd.DataFrame:
    candidates = _read(CANDIDATES_FILE)
    hist = _read(HISTORY_FILE)
    if candidates.empty or hist.empty:
        out = pd.DataFrame([{
            "as_of": pd.Timestamp.now().normalize(), "candidate_rows": 0, "sessions": 0,
            "baseline_close_mape_pct": np.nan, "adjusted_close_mape_pct": np.nan,
            "relative_mape_improvement_pct": np.nan, "baseline_direction_accuracy_pct": np.nan,
            "adjusted_direction_accuracy_pct": np.nan, "baseline_profitable_close_pct": np.nan,
            "adjusted_profitable_close_pct": np.nan, "baseline_mean_return_pct": np.nan,
            "adjusted_mean_return_pct": np.nan, "minimum_rows_required": MIN_ROWS,
            "minimum_sessions_required": MIN_SESSIONS,
            "minimum_relative_mape_improvement_pct": MIN_IMPROVEMENT * 100,
            "recent_error_promotion_evidence": False, "promotion_evidence": False,
            "validation_status": "collecting",
        }])
        out.to_csv(OUTPUT_FILE, index=False)
        return out

    c = candidates.copy()
    h = hist.copy()
    c["target_date"] = pd.to_datetime(c["target_date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    h["close"] = pd.to_numeric(h["close"], errors="coerce")
    h = h[["date", "symbol", "close"]].dropna().drop_duplicates(["date", "symbol"], keep="last")

    actual = h.rename(columns={"date": "target_date", "close": "actual_close"})
    x = c.merge(actual, on=["target_date", "symbol"], how="inner")
    for col in ["base_close", "predicted_close"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["selected"] = pd.to_numeric(x.get("selected"), errors="coerce").fillna(0).astype(int)
    x["rank"] = pd.to_numeric(x.get("rank"), errors="coerce")
    x = x.dropna(subset=["target_date", "base_close", "predicted_close", "actual_close", "rank"])
    x = x[x["base_close"] > 0]
    if x.empty:
        return pd.DataFrame()

    x["abs_error_pct"] = (x["predicted_close"] - x["actual_close"]).abs() / x["actual_close"] * 100.0
    x["direction_correct"] = (
        (x["predicted_close"] - x["base_close"]) * (x["actual_close"] - x["base_close"]) > 0
    ).astype(float)
    x["actual_return_pct"] = (x["actual_close"] / x["base_close"] - 1.0) * 100.0
    x["baseline"] = x["rank"] <= 10

    rows = []
    for target_date, g in x.groupby("target_date", sort=True):
        baseline = g[g["baseline"]]
        adjusted = g[g["selected"] == 1]
        if len(baseline) < 10 or len(adjusted) < 10:
            continue
        rows.append({
            "target_date": target_date,
            "baseline_rows": len(baseline),
            "adjusted_rows": len(adjusted),
            "baseline_close_mape_pct": baseline["abs_error_pct"].mean(),
            "adjusted_close_mape_pct": adjusted["abs_error_pct"].mean(),
            "baseline_direction_accuracy_pct": baseline["direction_correct"].mean() * 100.0,
            "adjusted_direction_accuracy_pct": adjusted["direction_correct"].mean() * 100.0,
            "baseline_profitable_close_pct": (baseline["actual_return_pct"] > 0).mean() * 100.0,
            "adjusted_profitable_close_pct": (adjusted["actual_return_pct"] > 0).mean() * 100.0,
            "baseline_mean_return_pct": baseline["actual_return_pct"].mean(),
            "adjusted_mean_return_pct": adjusted["actual_return_pct"].mean(),
        })

    sessions = pd.DataFrame(rows)
    if sessions.empty:
        summary = {
            "as_of": pd.Timestamp.now().normalize(), "candidate_rows": 0, "sessions": 0,
            "baseline_close_mape_pct": np.nan, "adjusted_close_mape_pct": np.nan,
            "relative_mape_improvement_pct": np.nan,
            "baseline_direction_accuracy_pct": np.nan, "adjusted_direction_accuracy_pct": np.nan,
            "baseline_profitable_close_pct": np.nan, "adjusted_profitable_close_pct": np.nan,
            "baseline_mean_return_pct": np.nan, "adjusted_mean_return_pct": np.nan,
            "minimum_rows_required": MIN_ROWS, "minimum_sessions_required": MIN_SESSIONS,
            "minimum_relative_mape_improvement_pct": MIN_IMPROVEMENT * 100,
            "recent_error_promotion_evidence": False, "promotion_evidence": False,
            "validation_status": "collecting",
        }
    else:
        baseline_mape = sessions["baseline_close_mape_pct"].mean()
        adjusted_mape = sessions["adjusted_close_mape_pct"].mean()
        rel = ((baseline_mape - adjusted_mape) / baseline_mape) if baseline_mape > 0 else np.nan
        rows_available = len(sessions) * 10
        evidence = bool(
            len(sessions) >= MIN_SESSIONS and rows_available >= MIN_ROWS
            and np.isfinite(rel) and rel >= MIN_IMPROVEMENT
            and sessions["adjusted_direction_accuracy_pct"].mean() >= sessions["baseline_direction_accuracy_pct"].mean()
            and sessions["adjusted_profitable_close_pct"].mean() >= sessions["baseline_profitable_close_pct"].mean()
            and sessions["adjusted_mean_return_pct"].mean() >= sessions["baseline_mean_return_pct"].mean()
        )
        summary = {
            "as_of": pd.Timestamp.now().normalize(), "candidate_rows": rows_available, "sessions": len(sessions),
            "baseline_close_mape_pct": baseline_mape, "adjusted_close_mape_pct": adjusted_mape,
            "relative_mape_improvement_pct": rel * 100.0 if np.isfinite(rel) else np.nan,
            "baseline_direction_accuracy_pct": sessions["baseline_direction_accuracy_pct"].mean(),
            "adjusted_direction_accuracy_pct": sessions["adjusted_direction_accuracy_pct"].mean(),
            "baseline_profitable_close_pct": sessions["baseline_profitable_close_pct"].mean(),
            "adjusted_profitable_close_pct": sessions["adjusted_profitable_close_pct"].mean(),
            "baseline_mean_return_pct": sessions["baseline_mean_return_pct"].mean(),
            "adjusted_mean_return_pct": sessions["adjusted_mean_return_pct"].mean(),
            "minimum_rows_required": MIN_ROWS, "minimum_sessions_required": MIN_SESSIONS,
            "minimum_relative_mape_improvement_pct": MIN_IMPROVEMENT * 100,
            "recent_error_promotion_evidence": evidence, "promotion_evidence": evidence,
            "validation_status": "promote" if evidence else ("collecting" if len(sessions) < MIN_SESSIONS else "hold"),
        }

    out = pd.DataFrame([summary])
    out.to_csv(OUTPUT_FILE, index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run_selection_validation()
