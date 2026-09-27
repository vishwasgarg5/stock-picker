from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVALUATIONS = DATA / "evaluations.csv"
OUTPUT = DATA / "ranking_validation.csv"

RANK_BUCKETS = [0, 10, 20, 30, 50, 75, 100, 125, 150]
RANK_LABELS = ["1-10", "11-20", "21-30", "31-50", "51-75", "76-100", "101-125", "126-150"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "target_date", "rank_bucket", "rows", "symbols", "mean_rank",
        "mean_score", "mean_close_return_pct", "median_close_return_pct",
        "profitable_close_pct", "close_mape_pct", "baseline_close_mape_pct",
        "mape_improvement_pct", "direction_accuracy_pct", "top10",
    ])


def run_ranking_diagnostics() -> pd.DataFrame:
    if not EVALUATIONS.exists():
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        print("Ranking diagnostics skipped: evaluations.csv missing.")
        return out

    x = pd.read_csv(EVALUATIONS)
    required = {
        "target_date", "symbol", "rank", "score", "base_close",
        "actual_open", "actual_close", "close_abs_pct_error",
        "baseline_close_abs_pct_error", "close_direction_correct",
    }
    if not required.issubset(x.columns):
        missing = sorted(required - set(x.columns))
        raise RuntimeError(f"Ranking diagnostics missing evaluation columns: {missing}")

    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    for col in ["rank", "score", "base_close", "actual_open", "actual_close",
                "close_abs_pct_error", "baseline_close_abs_pct_error",
                "close_direction_correct"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["target_date", "symbol", "rank", "score", "base_close", "actual_open", "actual_close"])
    if x.empty:
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        print("Ranking diagnostics skipped: no usable evaluations.")
        return out

    x["close_return_pct"] = (x["actual_close"] / x["actual_open"] - 1.0) * 100.0
    x["profitable_close"] = (x["close_return_pct"] > 0).astype(int)
    x["rank_bucket"] = pd.cut(
        x["rank"], bins=RANK_BUCKETS, labels=RANK_LABELS,
        include_lowest=True, right=True
    )
    x["top10"] = (x["rank"] <= 10).astype(int)

    out = (
        x.groupby(["target_date", "rank_bucket"], observed=False)
        .agg(
            rows=("symbol", "count"),
            symbols=("symbol", "nunique"),
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
            mean_close_return_pct=("close_return_pct", "mean"),
            median_close_return_pct=("close_return_pct", "median"),
            profitable_close_pct=("profitable_close", "mean"),
            close_mape_pct=("close_abs_pct_error", "mean"),
            baseline_close_mape_pct=("baseline_close_abs_pct_error", "mean"),
            direction_accuracy_pct=("close_direction_correct", "mean"),
            top10=("top10", "max"),
        )
        .reset_index()
    )
    out["profitable_close_pct"] *= 100.0
    out["close_mape_pct"] *= 100.0
    out["baseline_close_mape_pct"] *= 100.0
    out["direction_accuracy_pct"] *= 100.0
    out["mape_improvement_pct"] = np.where(
        out["baseline_close_mape_pct"].abs() > 1e-12,
        (out["baseline_close_mape_pct"] - out["close_mape_pct"]) / out["baseline_close_mape_pct"] * 100.0,
        np.nan,
    )

    # Add a compact session-level comparison of production Top-10 against ranks 11-20.
    top = x[x["rank"] <= 10].groupby("target_date").agg(
        top10_rows=("symbol", "count"),
        top10_score=("score", "mean"),
        top10_return_pct=("close_return_pct", "mean"),
        top10_profitable_pct=("profitable_close", lambda s: s.mean() * 100.0),
        top10_mape_pct=("close_abs_pct_error", lambda s: s.mean() * 100.0),
    )
    next10 = x[(x["rank"] >= 11) & (x["rank"] <= 20)].groupby("target_date").agg(
        next10_rows=("symbol", "count"),
        next10_score=("score", "mean"),
        next10_return_pct=("close_return_pct", "mean"),
        next10_profitable_pct=("profitable_close", lambda s: s.mean() * 100.0),
        next10_mape_pct=("close_abs_pct_error", lambda s: s.mean() * 100.0),
    )
    session = top.join(next10, how="outer").reset_index()
    session["top10_vs_next10_return_diff_pct"] = session["top10_return_pct"] - session["next10_return_pct"]
    session["top10_vs_next10_mape_diff_pct"] = session["top10_mape_pct"] - session["next10_mape_pct"]
    session["top10_outperformed_next10"] = (
        session["top10_return_pct"] > session["next10_return_pct"]
    ).astype("Int64")

    # Keep both bucket-level diagnostics and session Top-10/11-20 diagnostics
    # in one CSV; rows with rank_bucket identify the bucket records.
    session["rank_bucket"] = "TOP10_VS_11_20"
    session["rows"] = session["top10_rows"]
    session["symbols"] = session["top10_rows"]
    session["mean_rank"] = np.nan
    session["mean_score"] = session["top10_score"]
    session["mean_close_return_pct"] = session["top10_return_pct"]
    session["median_close_return_pct"] = np.nan
    session["profitable_close_pct"] = session["top10_profitable_pct"]
    session["close_mape_pct"] = session["top10_mape_pct"]
    session["baseline_close_mape_pct"] = np.nan
    session["mape_improvement_pct"] = np.nan
    session["direction_accuracy_pct"] = np.nan
    session["top10"] = 1
    session["target_date"] = pd.to_datetime(session["target_date"]).dt.normalize()
    session = session[out.columns.tolist()]

    out = pd.concat([out, session], ignore_index=True)
    out = out.sort_values(["target_date", "rank_bucket"]).reset_index(drop=True)
    out.to_csv(OUTPUT, index=False)

    latest = x["target_date"].max().date()
    latest_top = x[x["target_date"] == pd.Timestamp(latest)]
    latest_top10 = latest_top[latest_top["rank"] <= 10]
    latest_next10 = latest_top[(latest_top["rank"] >= 11) & (latest_top["rank"] <= 20)]
    print(
        f"Ranking diagnostics complete: sessions={x['target_date'].nunique()}, "
        f"evaluated_rows={len(x)}, latest={latest}, "
        f"top10_return={latest_top10['close_return_pct'].mean() * 100 / 100:.3f}%, "
        f"11-20_return={latest_next10['close_return_pct'].mean() * 100 / 100:.3f}%"
    )
    return out


if __name__ == "__main__":
    run_ranking_diagnostics()
