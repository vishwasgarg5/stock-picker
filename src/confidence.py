from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES = DATA / "prediction_candidates_history.csv"
HISTORY = DATA / "ohlcv.csv"
OUTPUT = DATA / "confidence_analysis.csv"
SELECTION_OUTPUT = DATA / "selection_validation.csv"
MIN_ROWS = 100
MIN_SESSIONS = 12
MIN_RELATIVE_IMPROVEMENT = 0.01

def run_confidence_analysis() -> pd.DataFrame:
    if not CANDIDATES.exists() or not HISTORY.exists():
        summary = pd.DataFrame([{"as_of": pd.NaT, "candidate_rows": 0, "sessions": 0, "baseline_close_mape_pct": np.nan, "confidence_close_mape_pct": np.nan, "relative_mape_improvement": np.nan, "baseline_direction_accuracy_pct": np.nan, "confidence_direction_accuracy_pct": np.nan, "baseline_profitable_close_pct": np.nan, "confidence_profitable_close_pct": np.nan, "minimum_rows_required": MIN_ROWS, "minimum_sessions_required": MIN_SESSIONS, "minimum_relative_mape_improvement": MIN_RELATIVE_IMPROVEMENT, "promotion_evidence": False, "validation_status": "collecting"}])
        summary.to_csv(SELECTION_OUTPUT, index=False)
        pd.DataFrame().to_csv(OUTPUT, index=False)
        return pd.DataFrame()
    c = pd.read_csv(CANDIDATES)
    h = pd.read_csv(HISTORY, parse_dates=["date"])
    for col in ["prediction_date", "target_date"]:
        c[col] = pd.to_datetime(c[col], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    actual = h.rename(columns={"date":"target_date","open":"actual_open","high":"actual_high","low":"actual_low","close":"actual_close"})
    x = c.merge(actual[["symbol","target_date","actual_open","actual_high","actual_low","actual_close"]], on=["symbol","target_date"], how="inner")
    if x.empty:
        summary = pd.DataFrame([{"as_of": pd.NaT, "candidate_rows": 0, "sessions": 0, "baseline_close_mape_pct": np.nan, "confidence_close_mape_pct": np.nan, "relative_mape_improvement": np.nan, "baseline_direction_accuracy_pct": np.nan, "confidence_direction_accuracy_pct": np.nan, "baseline_profitable_close_pct": np.nan, "confidence_profitable_close_pct": np.nan, "minimum_rows_required": MIN_ROWS, "minimum_sessions_required": MIN_SESSIONS, "minimum_relative_mape_improvement": MIN_RELATIVE_IMPROVEMENT, "promotion_evidence": False, "validation_status": "collecting"}])
        summary.to_csv(SELECTION_OUTPUT, index=False)
        pd.DataFrame().to_csv(OUTPUT, index=False)
        return pd.DataFrame()
    x["close_error_pct"] = (x["actual_close"]-x["predicted_close"]).abs()/x["actual_close"].abs()
    predicted_return = x["predicted_close"]/x["base_close"]-1
    actual_return = x["actual_close"]/x["base_close"]-1
    x["direction_correct"] = (np.sign(predicted_return)==np.sign(actual_return)).astype(int)
    x["mfe"] = x["actual_high"]/x["actual_open"]-1
    x["mae"] = x["actual_low"]/x["actual_open"]-1
    x["profitable_close"] = (x["actual_close"]/x["actual_open"]-1 > 0).astype(int)
    x["confidence_bucket"] = pd.cut(x["confidence_score"], bins=[-0.01,25,50,75,100.01], labels=["0-25","25-50","50-75","75-100"])
    out = x.groupby("confidence_bucket", observed=False).agg(
        rows=("symbol","count"), sessions=("target_date","nunique"),
        close_mape_pct=("close_error_pct", lambda s:s.mean()*100),
        direction_accuracy_pct=("direction_correct", lambda s:s.mean()*100),
        mfe_pct=("mfe", lambda s:s.mean()*100),
        mae_pct=("mae", lambda s:s.mean()*100),
        profitable_close_pct=("profitable_close", lambda s:s.mean()*100)
    ).reset_index()
    out["as_of"] = x["target_date"].max()
    out["minimum_rows_required"] = MIN_ROWS
    out["minimum_sessions_required"] = MIN_SESSIONS
    out["validation_status"] = np.where((out["rows"]>=MIN_ROWS)&(out["sessions"]>=MIN_SESSIONS),"validated_sample","collecting")
    # Promotion evidence must show a useful confidence gradient, not merely
    # a large sample. Require both low- and high-confidence buckets to have
    # sufficient data and the high-confidence bucket to have lower error.
    low = out[out["confidence_bucket"] == "0-25"]
    high = out[out["confidence_bucket"] == "75-100"]
    evidence = bool(
        not low.empty and not high.empty
        and low.iloc[0]["validation_status"] == "validated_sample"
        and high.iloc[0]["validation_status"] == "validated_sample"
        and high.iloc[0]["close_mape_pct"] < low.iloc[0]["close_mape_pct"]
    )
    out["promotion_evidence"] = evidence
    out.to_csv(OUTPUT,index=False)

    # Directly backtest the production decision rule against the existing
    # ranking Top-10 baseline, session by session. This is deliberately
    # based only on information available at prediction time.
    session_rows = []
    for target_date, g in x.groupby("target_date"):
        g = g.dropna(subset=["rank", "confidence_score", "close_error_pct"]).copy()
        if len(g) < 10:
            continue
        baseline = g.sort_values(["rank", "symbol"]).head(10).copy()
        g["selection_priority"] = g["rank"] + ((100.0 - g["confidence_score"]) / 100.0)
        challenger = g.sort_values(["selection_priority", "rank", "symbol"]).head(10).copy()

        def metrics(z):
            return {
                "mape": z["close_error_pct"].mean() * 100.0,
                "direction": z["direction_correct"].mean() * 100.0,
                "profitable": z["profitable_close"].mean() * 100.0,
                "close_return": (z["actual_close"] / z["actual_open"] - 1.0).mean() * 100.0,
            }

        b = metrics(baseline)
        q = metrics(challenger)
        session_rows.append({
            "target_date": target_date,
            "baseline_mape_pct": b["mape"],
            "confidence_mape_pct": q["mape"],
            "baseline_direction_accuracy_pct": b["direction"],
            "confidence_direction_accuracy_pct": q["direction"],
            "baseline_profitable_close_pct": b["profitable"],
            "confidence_profitable_close_pct": q["profitable"],
            "baseline_equal_weight_close_return_pct": b["close_return"],
            "confidence_equal_weight_close_return_pct": q["close_return"],
            "confidence_improved_mape": int(q["mape"] < b["mape"]),
        })

    validation = pd.DataFrame(session_rows)
    if validation.empty:
        validation = pd.DataFrame(columns=[
            "target_date","baseline_mape_pct","confidence_mape_pct",
            "baseline_direction_accuracy_pct","confidence_direction_accuracy_pct",
            "baseline_profitable_close_pct","confidence_profitable_close_pct",
            "baseline_equal_weight_close_return_pct",
            "confidence_equal_weight_close_return_pct","confidence_improved_mape",
        ])

    sessions = len(validation)
    rows = len(x)
    if sessions:
        baseline_mape = validation["baseline_mape_pct"].mean()
        confidence_mape = validation["confidence_mape_pct"].mean()
        relative_improvement = (
            (baseline_mape - confidence_mape) / baseline_mape
            if baseline_mape and np.isfinite(baseline_mape) else np.nan
        )
        baseline_direction = validation["baseline_direction_accuracy_pct"].mean()
        confidence_direction = validation["confidence_direction_accuracy_pct"].mean()
        baseline_profit = validation["baseline_profitable_close_pct"].mean()
        confidence_profit = validation["confidence_profitable_close_pct"].mean()
        gate = bool(
            rows >= MIN_ROWS and sessions >= MIN_SESSIONS
            and np.isfinite(relative_improvement)
            and relative_improvement >= MIN_RELATIVE_IMPROVEMENT
            and confidence_direction >= baseline_direction
            and confidence_profit >= baseline_profit
        )
    else:
        baseline_mape = confidence_mape = relative_improvement = np.nan
        baseline_direction = confidence_direction = np.nan
        baseline_profit = confidence_profit = np.nan
        gate = False

    summary = pd.DataFrame([{
        "as_of": x["target_date"].max(),
        "candidate_rows": rows,
        "sessions": sessions,
        "baseline_close_mape_pct": baseline_mape,
        "confidence_close_mape_pct": confidence_mape,
        "relative_mape_improvement": relative_improvement,
        "baseline_direction_accuracy_pct": baseline_direction,
        "confidence_direction_accuracy_pct": confidence_direction,
        "baseline_profitable_close_pct": baseline_profit,
        "confidence_profitable_close_pct": confidence_profit,
        "minimum_rows_required": MIN_ROWS,
        "minimum_sessions_required": MIN_SESSIONS,
        "minimum_relative_mape_improvement": MIN_RELATIVE_IMPROVEMENT,
        "promotion_evidence": gate,
        "validation_status": "validated" if gate else ("collecting" if rows < MIN_ROWS or sessions < MIN_SESSIONS else "not_validated"),
    }])
    summary.to_csv(SELECTION_OUTPUT, index=False)
    return out

if __name__ == "__main__":
    result=run_confidence_analysis()
    print(f"Confidence analysis complete: {len(result)} buckets")
