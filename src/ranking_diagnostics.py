from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES = DATA / "prediction_candidates_history.csv"
HISTORY = DATA / "ohlcv.csv"
OUTPUT = DATA / "ranking_validation.csv"

RANK_BUCKETS = [0, 10, 20]
RANK_LABELS = ["1-10", "11-20"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "target_date", "rank_bucket", "rows", "symbols", "mean_rank",
        "mean_score", "mean_close_return_pct", "median_close_return_pct",
        "profitable_close_pct", "top10", "top10_vs_next10_return_diff_pct",
        "top10_vs_next10_profitable_diff_pct", "top10_outperformed_next10",
    ])


def _next_session_actuals(history: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    h = history[history["date"] >= target_date].sort_values(["symbol", "date"])
    return h.groupby("symbol", as_index=False).first()[["symbol", "date", "open", "close"]].rename(
        columns={"date": "actual_date", "open": "actual_open", "close": "actual_close"}
    )


def run_ranking_diagnostics() -> pd.DataFrame:
    if not CANDIDATES.exists() or not HISTORY.exists():
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        print("Ranking diagnostics skipped: candidate history or OHLCV history missing.")
        return out

    c = pd.read_csv(CANDIDATES)
    h = pd.read_csv(HISTORY, parse_dates=["date"])
    c["target_date"] = pd.to_datetime(c["target_date"], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    h["symbol"] = h["symbol"].astype(str).str.strip()

    for col in ["rank", "score"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["target_date", "symbol", "rank", "score"])
    c = c[c["rank"].between(1, 20)].copy()
    if c.empty:
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        print("Ranking diagnostics skipped: no usable Top-20 candidates.")
        return out

    actual_parts = []
    for target_date in sorted(c["target_date"].dropna().unique()):
        actual = _next_session_actuals(h, pd.Timestamp(target_date))
        actual["target_date"] = pd.Timestamp(target_date)
        actual_parts.append(actual)
    actuals = pd.concat(actual_parts, ignore_index=True) if actual_parts else pd.DataFrame()

    x = c.merge(actuals, on=["target_date", "symbol"], how="left")
    x = x.dropna(subset=["actual_open", "actual_close"])
    if x.empty:
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        print("Ranking diagnostics skipped: no matching next-session OHLC data.")
        return out

    x["close_return_pct"] = (x["actual_close"] / x["actual_open"] - 1.0) * 100.0
    x["profitable_close"] = (x["close_return_pct"] > 0).astype(int)
    x["rank_bucket"] = pd.cut(
        x["rank"], bins=RANK_BUCKETS, labels=RANK_LABELS,
        include_lowest=True, right=True
    )
    x["top10"] = (x["rank"] <= 10).astype(int)

    bucket = (
        x.groupby(["target_date", "rank_bucket"], observed=False)
        .agg(
            rows=("symbol", "count"),
            symbols=("symbol", "nunique"),
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
            mean_close_return_pct=("close_return_pct", "mean"),
            median_close_return_pct=("close_return_pct", "median"),
            profitable_close_pct=("profitable_close", lambda s: s.mean() * 100.0),
            top10=("top10", "max"),
        )
        .reset_index()
    )

    comparisons = []
    for target_date, g in x.groupby("target_date"):
        top = g[g["rank"] <= 10]
        nxt = g[g["rank"].between(11, 20)]
        if top.empty or nxt.empty:
            continue
        top_ret = top["close_return_pct"].mean()
        next_ret = nxt["close_return_pct"].mean()
        top_prof = top["profitable_close"].mean() * 100.0
        next_prof = nxt["profitable_close"].mean() * 100.0
        comparisons.append({
            "target_date": target_date,
            "rank_bucket": "TOP10_VS_11_20",
            "rows": len(top),
            "symbols": top["symbol"].nunique(),
            "mean_rank": top["rank"].mean(),
            "mean_score": top["score"].mean(),
            "mean_close_return_pct": top_ret,
            "median_close_return_pct": top["close_return_pct"].median(),
            "profitable_close_pct": top_prof,
            "top10": 1,
            "top10_vs_next10_return_diff_pct": top_ret - next_ret,
            "top10_vs_next10_profitable_diff_pct": top_prof - next_prof,
            "top10_outperformed_next10": int(top_ret > next_ret),
        })

    out = bucket.copy()
    for col in [
        "top10_vs_next10_return_diff_pct",
        "top10_vs_next10_profitable_diff_pct",
        "top10_outperformed_next10",
    ]:
        out[col] = np.nan
    out = pd.concat([out, pd.DataFrame(comparisons)], ignore_index=True)
    out = out.sort_values(["target_date", "rank_bucket"]).reset_index(drop=True)
    out.to_csv(OUTPUT, index=False)

    sessions = x["target_date"].nunique()
    top = x[x["rank"] <= 10]
    nxt = x[x["rank"].between(11, 20)]
    print(
        f"Ranking diagnostics complete: sessions={sessions}, evaluated_candidates={len(x)}, "
        f"top10_return={top['close_return_pct'].mean():.3f}%, "
        f"11-20_return={nxt['close_return_pct'].mean():.3f}%, "
        f"top10_profitable={top['profitable_close'].mean()*100:.1f}%, "
        f"11-20_profitable={nxt['profitable_close'].mean()*100:.1f}%"
    )
    return out


if __name__ == "__main__":
    run_ranking_diagnostics()
