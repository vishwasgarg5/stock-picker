from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES_FILE = DATA / "prediction_candidates_history.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_FILE = DATA / "selection_validation.csv"
PREDICTIONS_FILE = DATA / "predictions.csv"
COVERAGE_FILE = DATA / "selection_validation_coverage.json"

MIN_ROWS = 100
MIN_SESSIONS = 12
MIN_IMPROVEMENT = 0.01  # 1% relative MAPE improvement


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _write_coverage_audit(predictions: pd.DataFrame, candidates: pd.DataFrame, history: pd.DataFrame) -> dict:
    def session_dates(frame: pd.DataFrame) -> set[str]:
        if frame.empty or "prediction_date" not in frame.columns:
            return set()
        dates = pd.to_datetime(frame["prediction_date"], errors="coerce").dt.normalize()
        return set(dates.dropna().dt.strftime("%Y-%m-%d"))

    prediction_sessions = session_dates(predictions)
    candidate_sessions = session_dates(candidates)
    missing_sessions = sorted(prediction_sessions - candidate_sessions)
    duplicate_keys = 0
    matched_rows = 0
    unmatched_rows = len(candidates)
    if not candidates.empty and {"prediction_date", "target_date", "symbol"}.issubset(candidates.columns):
        keys = ["prediction_date", "target_date", "symbol"]
        normalized = candidates.copy()
        normalized["prediction_date"] = pd.to_datetime(normalized["prediction_date"], errors="coerce").dt.normalize()
        normalized["target_date"] = pd.to_datetime(normalized["target_date"], errors="coerce").dt.normalize()
        normalized["symbol"] = normalized["symbol"].astype(str).str.upper().str.strip()
        duplicate_keys = int(normalized.duplicated(keys, keep=False).sum())
        if not history.empty and {"date", "symbol", "close"}.issubset(history.columns):
            actual = history[["date", "symbol", "close"]].copy()
            actual["target_date"] = pd.to_datetime(actual.pop("date"), errors="coerce").dt.normalize()
            actual["symbol"] = actual["symbol"].astype(str).str.upper().str.strip()
            actual["close"] = pd.to_numeric(actual["close"], errors="coerce")
            actual = actual.dropna(subset=["target_date", "symbol", "close"]).drop_duplicates(
                ["target_date", "symbol"], keep="last"
            )
            merged = normalized.merge(
                actual[["target_date", "symbol", "close"]],
                on=["target_date", "symbol"], how="left", validate="many_to_one"
            )
            matched_rows = int(merged["close"].notna().sum())
            unmatched_rows = int(merged["close"].isna().sum())
    result = {
        "status": "MATCHED_DATA_AVAILABLE" if matched_rows else "BLOCKED_NO_MATCHED_OUTCOMES",
        "prediction_rows": int(len(predictions)),
        "prediction_sessions": len(prediction_sessions),
        "prediction_session_dates": sorted(prediction_sessions),
        "candidate_rows": int(len(candidates)),
        "candidate_sessions": len(candidate_sessions),
        "candidate_session_dates": sorted(candidate_sessions),
        "prediction_sessions_missing_candidate_history": missing_sessions,
        "missing_candidate_session_count": len(missing_sessions),
        "duplicate_candidate_vintage_keys": duplicate_keys,
        "history_rows": int(len(history)),
        "matched_candidate_rows": matched_rows,
        "candidate_rows_without_actual_close": unmatched_rows,
        "minimum_sessions_required": MIN_SESSIONS,
        "minimum_rows_required": MIN_ROWS,
        "candidate_history_reconstructed_from_top10_predictions": False,
        "missing_facts_inferred": False,
    }
    COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _blocked_summary(candidate_rows: int, status: str, reason: str) -> pd.DataFrame:
    summary = pd.DataFrame([{
        "as_of": pd.Timestamp.now().normalize(),
        "candidate_rows": int(candidate_rows),
        "sessions": 0,
        "baseline_close_mape_pct": np.nan,
        "adjusted_close_mape_pct": np.nan,
        "relative_mape_improvement_pct": np.nan,
        "baseline_direction_accuracy_pct": np.nan,
        "adjusted_direction_accuracy_pct": np.nan,
        "baseline_profitable_close_pct": np.nan,
        "adjusted_profitable_close_pct": np.nan,
        "baseline_mean_return_pct": np.nan,
        "adjusted_mean_return_pct": np.nan,
        "minimum_rows_required": MIN_ROWS,
        "minimum_sessions_required": MIN_SESSIONS,
        "minimum_relative_mape_improvement_pct": MIN_IMPROVEMENT * 100,
        "recent_error_promotion_evidence": False,
        "confidence_promotion_evidence": False,
        "promotion_evidence": False,
        "validation_status": status,
        "validation_block_reason": reason,
    }])
    summary.to_csv(OUTPUT_FILE, index=False)
    return summary


def run_selection_validation() -> pd.DataFrame:
    candidates = _read(CANDIDATES_FILE)
    hist = _read(HISTORY_FILE)
    predictions = _read(PREDICTIONS_FILE)
    _write_coverage_audit(predictions, candidates, hist)
    if candidates.empty or hist.empty:
        if candidates.empty:
            reason = "Candidate history is empty; top-10 prediction rows are not a substitute for the full candidate pool"
            status = "blocked_missing_candidate_history"
        else:
            reason = "Realized OHLC history is missing or empty; candidate outcomes cannot be matched"
            status = "blocked_missing_actual_history"
        return _blocked_summary(len(candidates), status, reason)

    c = candidates.copy()
    h = hist.copy()
    c["target_date"] = pd.to_datetime(c["target_date"], errors="coerce").dt.normalize()
    if "prediction_date" not in c.columns:
        return _blocked_summary(len(c), "blocked_missing_prediction_date",
                                "Forecast vintage key is missing; no selection evidence was scored")
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
    candidate_keys = ["prediction_date", "target_date", "symbol"]
    if c.duplicated(candidate_keys).any():
        duplicate_rows = int(c.duplicated(candidate_keys, keep=False).sum())
        return _blocked_summary(len(c), "blocked_duplicate_candidate_keys",
                                f"{duplicate_rows} rows have duplicate forecast-vintage keys")
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    h["close"] = pd.to_numeric(h["close"], errors="coerce")
    h = h[["date", "symbol", "close"]].dropna().drop_duplicates(["date", "symbol"], keep="last")

    actual = h.rename(columns={"date": "target_date", "close": "actual_close"})
    x = c.merge(actual, on=["target_date", "symbol"], how="left", validate="many_to_one")
    for col in ["base_close", "predicted_close"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["selected"] = pd.to_numeric(x.get("selected"), errors="coerce").fillna(0).astype(int)
    x["rank"] = pd.to_numeric(x.get("rank"), errors="coerce")
    x = x.dropna(subset=["prediction_date", "target_date", "base_close", "predicted_close", "actual_close", "rank"])
    x = x[x["base_close"] > 0]
    if x.empty:
        return _blocked_summary(len(c), "blocked_no_matched_outcomes",
                                "No candidate rows matched realized OHLC outcomes; prior promotion evidence was cleared")

    x["abs_error_pct"] = (x["predicted_close"] - x["actual_close"]).abs() / x["actual_close"] * 100.0
    x["direction_correct"] = (
        (x["predicted_close"] - x["base_close"]) * (x["actual_close"] - x["base_close"]) > 0
    ).astype(float)
    x["actual_return_pct"] = (x["actual_close"] / x["base_close"] - 1.0) * 100.0
    x["baseline"] = x["rank"] <= 10

    rows = []
    # Score each forecast vintage independently. A symbol can have multiple
    # forecasts for the same target session; collapsing by target_date alone
    # would double-weight or overwrite one prediction session with another.
    for (prediction_date, target_date), g in x.groupby(["prediction_date", "target_date"], sort=True):
        baseline = g[g["baseline"]]
        adjusted = g[g["selected"] == 1]
        if len(baseline) < 10 or len(adjusted) < 10:
            continue
        rows.append({
            "prediction_date": prediction_date,
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
