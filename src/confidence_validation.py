from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_FILE = DATA / "confidence_validation.csv"

MIN_ROWS = 100
MIN_SESSIONS = 12
MIN_TOP_BOTTOM_IMPROVEMENT = 0.05


def run_confidence_validation() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists() or not HISTORY_FILE.exists():
        return _write([{
            "as_of": pd.Timestamp.now().normalize(), "rows": 0, "sessions": 0,
            "top_confidence_mape_pct": np.nan, "bottom_confidence_mape_pct": np.nan,
            "top_bottom_mape_improvement_pct": np.nan,
            "top_confidence_direction_pct": np.nan, "bottom_confidence_direction_pct": np.nan,
            "confidence_promotion_evidence": False, "validation_status": "collecting",
        }])

    p = pd.read_csv(PREDICTIONS_FILE, parse_dates=["target_date"])
    h = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    if p.empty or h.empty or "prediction_spread" not in p.columns:
        return _write([{
            "as_of": pd.Timestamp.now().normalize(), "rows": 0, "sessions": 0,
            "confidence_promotion_evidence": False, "validation_status": "collecting",
        }])

    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    p["prediction_spread"] = pd.to_numeric(p["prediction_spread"], errors="coerce")
    p["base_close"] = pd.to_numeric(p["base_close"], errors="coerce")
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    h["close"] = pd.to_numeric(h["close"], errors="coerce")

    actual = h[["date", "symbol", "close"]].rename(
        columns={"date": "target_date", "close": "actual_close"}
    )
    x = p.merge(actual, on=["target_date", "symbol"], how="inner")
    x = x.dropna(subset=["target_date", "prediction_spread", "base_close", "actual_close"])
    x = x[x["base_close"] > 0]
    if x.empty:
        return _write([{
            "as_of": pd.Timestamp.now().normalize(), "rows": 0, "sessions": 0,
            "confidence_promotion_evidence": False, "validation_status": "collecting",
        }])

    x["close_abs_error_pct"] = (
        (x["actual_close"] - pd.to_numeric(x["predicted_close"], errors="coerce")).abs()
        / x["actual_close"].abs().replace(0, np.nan) * 100.0
    )
    x["direction_correct"] = (
        (pd.to_numeric(x["predicted_close"], errors="coerce") - x["base_close"])
        * (x["actual_close"] - x["base_close"]) > 0
    ).astype(int)

    # Use within-session percentile to avoid one unusually volatile market day
    # dominating the confidence calibration.
    # Calibrate confidence from lower ensemble uncertainty, not from the size of the predicted move.\n    x["confidence_score_calibrated"] = (1.0 - x.groupby("target_date")["prediction_spread"].rank(pct=True, method="average")).clip(0.0, 1.0) * 100.0\n    x["confidence_pct"] = x.groupby("target_date")["confidence_score_calibrated"].rank(pct=True, method="average")
    x["confidence_bucket"] = pd.cut(
        x["confidence_pct"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        include_lowest=True,
    )

    buckets = x.groupby("confidence_bucket", observed=False).agg(
        rows=("symbol", "size"),
        mape_pct=("close_abs_error_pct", "mean"),
        direction_accuracy_pct=("direction_correct", "mean"),
    ).reset_index()
    buckets["direction_accuracy_pct"] *= 100.0

    top = x[x["confidence_pct"] >= 0.8]
    bottom = x[x["confidence_pct"] <= 0.2]
    top_mape = float(top["close_abs_error_pct"].mean()) if not top.empty else np.nan
    bottom_mape = float(bottom["close_abs_error_pct"].mean()) if not bottom.empty else np.nan
    top_dir = float(top["direction_correct"].mean() * 100.0) if not top.empty else np.nan
    bottom_dir = float(bottom["direction_correct"].mean() * 100.0) if not bottom.empty else np.nan
    improvement = ((bottom_mape - top_mape) / bottom_mape) if bottom_mape > 0 else np.nan

    sessions = int(x["target_date"].nunique())
    rows = int(len(x))
    evidence = bool(
        rows >= MIN_ROWS and sessions >= MIN_SESSIONS
        and np.isfinite(improvement) and improvement >= MIN_TOP_BOTTOM_IMPROVEMENT
        and np.isfinite(top_dir) and np.isfinite(bottom_dir) and top_dir >= bottom_dir
    )

    summary = pd.DataFrame([{
        "as_of": pd.Timestamp.now().normalize(),
        "rows": rows,
        "sessions": sessions,
        "top_confidence_mape_pct": top_mape,
        "bottom_confidence_mape_pct": bottom_mape,
        "top_bottom_mape_improvement_pct": improvement * 100.0 if np.isfinite(improvement) else np.nan,
        "top_confidence_direction_pct": top_dir,
        "bottom_confidence_direction_pct": bottom_dir,
        "confidence_promotion_evidence": evidence,
        "validation_status": "promote" if evidence else ("collecting" if rows < MIN_ROWS or sessions < MIN_SESSIONS else "hold"),
    }])

    # Store both summary and bucket-level calibration so later trade filtering
    # can use empirical confidence bands rather than arbitrary thresholds.
    buckets["as_of"] = pd.Timestamp.now().normalize()
    buckets["summary_rows"] = rows
    buckets["summary_sessions"] = sessions
    buckets["confidence_promotion_evidence"] = evidence
    buckets.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(OUTPUT_FILE.with_name("confidence_validation_summary.csv"), index=False)
    return summary


def _write(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE.with_name("confidence_validation_summary.csv"), index=False)
    return out


if __name__ == "__main__":
    result = run_confidence_validation()
    print(result.to_string(index=False))
