from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES = DATA / "prediction_candidates_history.csv"
HISTORY = DATA / "ohlcv.csv"
OUTPUT = DATA / "ranking_validation.csv"
HORIZON_OUTPUT = DATA / "ranking_validation_horizons.csv"
FUNDAMENTALS_HISTORY = DATA / "fundamentals_history.csv"
SECTOR_OUTPUT = DATA / "ranking_validation_sector_risk.csv"

RANK_BUCKETS = [0, 10, 20]
RANK_LABELS = ["1-10", "11-20"]
HORIZONS = (1, 3, 5, 10, 20)


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


def _empty_horizons() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "horizon_sessions", "group", "rows", "symbols",
        "mean_return_pct", "median_return_pct", "hit_rate_pct",
        "universe_mean_return_pct", "lift_vs_universe_pct",
        "comparator_group", "comparator_mean_return_pct",
        "lift_vs_comparator_pct",
    ])


def _future_close(history: pd.DataFrame, prediction_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Return the close exactly horizon trading sessions after prediction_date."""
    h = history.sort_values(["symbol", "date"]).copy()
    h = h[h["date"] >= prediction_date]
    h["session_no"] = h.groupby("symbol").cumcount()
    out = h[h["session_no"] == horizon][["symbol", "date", "close"]].copy()
    return out.rename(columns={"date": "future_date", "close": "future_close"})


def _session_returns(history: pd.DataFrame, symbols: pd.Series, prediction_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Calculate close-to-close forward return using trading sessions."""
    wanted = set(symbols.astype(str))
    base = history[(history["date"] == prediction_date) & (history["symbol"].isin(wanted))][["symbol", "close"]].rename(columns={"close": "base_close"})
    future = _future_close(history, prediction_date, horizon)
    x = base.merge(future, on="symbol", how="inner")
    x["return_pct"] = (x["future_close"] / x["base_close"] - 1.0) * 100.0
    return x.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_pct"])


def run_multi_horizon_validation(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Validate ranking quality at 1/5/10/20 trading-session horizons."""
    rows = []
    if candidates.empty or history.empty:
        return _empty_horizons()
    c = candidates.copy()
    h = history.copy()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    h["symbol"] = h["symbol"].astype(str).str.strip()
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    c = c.dropna(subset=["prediction_date", "symbol", "rank"])
    c = c[c["rank"].between(1, 20)].copy()
    h = h.dropna(subset=["date", "symbol", "close"])

    for prediction_date in sorted(c["prediction_date"].unique()):
        d = pd.Timestamp(prediction_date)
        session_candidates = c[c["prediction_date"] == d]
        universe_symbols = h.loc[h["date"] == d, "symbol"].dropna().unique()
        if len(universe_symbols) == 0:
            continue
        for horizon in HORIZONS:
            universe = _session_returns(h, pd.Series(universe_symbols), d, horizon)
            if universe.empty:
                continue
            universe_mean = universe["return_pct"].mean()
            group_map = {
                "TOP5": session_candidates[session_candidates["rank"] <= 5],
                "TOP10": session_candidates[session_candidates["rank"] <= 10],
                "11-20": session_candidates[session_candidates["rank"].between(11, 20)],
                "TOP20": session_candidates[session_candidates["rank"] <= 20],
            }
            group_returns = {}
            for group, g in group_map.items():
                if g.empty:
                    continue
                r = _session_returns(h, g["symbol"], d, horizon)
                if not r.empty:
                    group_returns[group] = r
            comparator = group_returns.get("11-20")
            comparator_mean = comparator["return_pct"].mean() if comparator is not None else np.nan
            for group, r in group_returns.items():
                mean_ret = r["return_pct"].mean()
                rows.append({
                    "prediction_date": d,
                    "horizon_sessions": horizon,
                    "group": group,
                    "rows": len(r),
                    "symbols": r["symbol"].nunique(),
                    "mean_return_pct": mean_ret,
                    "median_return_pct": r["return_pct"].median(),
                    "hit_rate_pct": (r["return_pct"] > 0).mean() * 100.0,
                    "universe_mean_return_pct": universe_mean,
                    "lift_vs_universe_pct": mean_ret - universe_mean,
                    "comparator_group": "11-20" if group in {"TOP5", "TOP10"} else "",
                    "comparator_mean_return_pct": comparator_mean if group in {"TOP5", "TOP10"} else np.nan,
                    "lift_vs_comparator_pct": mean_ret - comparator_mean if group in {"TOP5", "TOP10"} and pd.notna(comparator_mean) else np.nan,
                })
            rows.append({
                "prediction_date": d,
                "horizon_sessions": horizon,
                "group": "UNIVERSE",
                "rows": len(universe),
                "symbols": universe["symbol"].nunique(),
                "mean_return_pct": universe_mean,
                "median_return_pct": universe["return_pct"].median(),
                "hit_rate_pct": (universe["return_pct"] > 0).mean() * 100.0,
                "universe_mean_return_pct": universe_mean,
                "lift_vs_universe_pct": 0.0,
                "comparator_group": "",
                "comparator_mean_return_pct": np.nan,
                "lift_vs_comparator_pct": np.nan,
            })
    if not rows:
        return _empty_horizons()
    return pd.DataFrame(rows).sort_values(["prediction_date", "horizon_sessions", "group"]).reset_index(drop=True)



def _empty_mfe_mae() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "horizon_sessions", "group", "rows", "symbols",
        "mean_mfe_pct", "median_mfe_pct", "mean_mae_pct", "median_mae_pct",
        "positive_mfe_pct", "mean_final_return_pct", "median_final_return_pct",
    ])


def _future_path(history: pd.DataFrame, prediction_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Return high/low/close for the next H trading sessions, excluding prediction_date."""
    h = history.sort_values(["symbol", "date"]).copy()
    h = h[h["date"] > prediction_date]
    h["session_no"] = h.groupby("symbol").cumcount() + 1
    return h[h["session_no"] <= horizon][["symbol", "date", "high", "low", "close"]].copy()


def _mfe_mae_returns(
    history: pd.DataFrame,
    symbols: pd.Series,
    prediction_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Calculate path MFE/MAE from prediction-date close through the next H sessions."""
    wanted = set(symbols.astype(str))
    base = history[
        (history["date"] == prediction_date) & history["symbol"].isin(wanted)
    ][["symbol", "close"]].rename(columns={"close": "base_close"})
    future = _future_path(history, prediction_date, horizon)
    if future.empty:
        return pd.DataFrame()

    x = base.merge(future, on="symbol", how="inner")
    if x.empty:
        return x

    for col in ["high", "low", "close"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["base_close"] = pd.to_numeric(x["base_close"], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["base_close", "high", "low", "close"])
    if x.empty:
        return x

    x["mfe_pct"] = (x["high"] / x["base_close"] - 1.0) * 100.0
    x["mae_pct"] = (x["low"] / x["base_close"] - 1.0) * 100.0
    final = (
        x.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol", "close"]]
        .rename(columns={"close": "final_close"})
    )
    x = x.merge(final, on="symbol", how="inner")
    x["final_return_pct"] = (x["final_close"] / x["base_close"] - 1.0) * 100.0
    return (
        x.groupby("symbol", as_index=False)
        .agg(
            mfe_pct=("mfe_pct", "max"),
            mae_pct=("mae_pct", "min"),
            final_return_pct=("final_return_pct", "last"),
        )
    )


def run_mfe_mae_validation(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Validate maximum favorable/adverse movement for ranking groups at each horizon."""
    if candidates.empty or history.empty:
        return _empty_mfe_mae()

    c = candidates.copy()
    h = history.copy()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    h["symbol"] = h["symbol"].astype(str).str.strip()
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    c = c.dropna(subset=["prediction_date", "symbol", "rank"])
    c = c[c["rank"].between(1, 20)].copy()
    h = h.dropna(subset=["date", "symbol", "close", "high", "low"])

    rows = []
    for prediction_date in sorted(c["prediction_date"].unique()):
        d = pd.Timestamp(prediction_date)
        session_candidates = c[c["prediction_date"] == d]
        universe_symbols = h.loc[h["date"] == d, "symbol"].dropna().unique()
        groups = {
            "TOP5": session_candidates[session_candidates["rank"] <= 5],
            "TOP10": session_candidates[session_candidates["rank"] <= 10],
            "11-20": session_candidates[session_candidates["rank"].between(11, 20)],
            "TOP20": session_candidates[session_candidates["rank"] <= 20],
            "UNIVERSE": pd.DataFrame({"symbol": universe_symbols}),
        }
        for horizon in HORIZONS:
            for group, g in groups.items():
                if g.empty:
                    continue
                r = _mfe_mae_returns(h, g["symbol"], d, horizon)
                if r.empty:
                    continue
                rows.append({
                    "prediction_date": d,
                    "horizon_sessions": horizon,
                    "group": group,
                    "rows": len(r),
                    "symbols": r["symbol"].nunique(),
                    "mean_mfe_pct": r["mfe_pct"].mean(),
                    "median_mfe_pct": r["mfe_pct"].median(),
                    "mean_mae_pct": r["mae_pct"].mean(),
                    "median_mae_pct": r["mae_pct"].median(),
                    "positive_mfe_pct": (r["mfe_pct"] > 0).mean() * 100.0,
                    "mean_final_return_pct": r["final_return_pct"].mean(),
                    "median_final_return_pct": r["final_return_pct"].median(),
                })

    if not rows:
        return _empty_mfe_mae()
    return pd.DataFrame(rows).sort_values(
        ["prediction_date", "horizon_sessions", "group"]
    ).reset_index(drop=True)



STABILITY_OUTPUT = DATA / "ranking_validation_stability.csv"


def _empty_stability() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "previous_prediction_date", "top5_overlap_pct",
        "top10_overlap_pct", "top20_overlap_pct", "top10_retention_pct",
        "mean_abs_rank_change", "median_abs_rank_change",
        "ranked_symbols", "previous_ranked_symbols",
    ])


def run_ranking_stability(candidates: pd.DataFrame) -> pd.DataFrame:
    """Measure consecutive-session ranking stability without using future outcomes."""
    if candidates.empty:
        return _empty_stability()

    c = candidates.copy()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    c = c.dropna(subset=["prediction_date", "symbol", "rank"])
    c = c[c["rank"].between(1, 20)].copy()

    sessions = sorted(c["prediction_date"].unique())
    rows = []
    for previous_date, current_date in zip(sessions, sessions[1:]):
        prev = c[c["prediction_date"] == previous_date].set_index("symbol")["rank"]
        curr = c[c["prediction_date"] == current_date].set_index("symbol")["rank"]

        def overlap_pct(n: int) -> float:
            a = set(prev[prev <= n].index)
            b = set(curr[curr <= n].index)
            return len(a & b) / min(len(a), len(b)) * 100.0 if a and b else np.nan

        common = prev.index.intersection(curr.index)
        rank_change = (curr.loc[common] - prev.loc[common]).abs()
        top10_prev = set(prev[prev <= 10].index)
        top10_curr = set(curr[curr <= 10].index)
        retained = len(top10_prev & top10_curr) / len(top10_prev) * 100.0 if top10_prev else np.nan

        rows.append({
            "prediction_date": pd.Timestamp(current_date),
            "previous_prediction_date": pd.Timestamp(previous_date),
            "top5_overlap_pct": overlap_pct(5),
            "top10_overlap_pct": overlap_pct(10),
            "top20_overlap_pct": overlap_pct(20),
            "top10_retention_pct": retained,
            "mean_abs_rank_change": rank_change.mean() if not rank_change.empty else np.nan,
            "median_abs_rank_change": rank_change.median() if not rank_change.empty else np.nan,
            "ranked_symbols": curr.index.nunique(),
            "previous_ranked_symbols": prev.index.nunique(),
        })

    if not rows:
        return _empty_stability()
    return pd.DataFrame(rows).sort_values("prediction_date").reset_index(drop=True)



TURNOVER_OUTPUT = DATA / "ranking_validation_turnover.csv"


def _empty_turnover() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "previous_prediction_date",
        "top5_turnover_pct", "top10_turnover_pct", "top20_turnover_pct",
        "top5_entries", "top5_exits", "top10_entries", "top10_exits",
        "top20_entries", "top20_exits",
    ])


def run_ranking_turnover(candidates: pd.DataFrame) -> pd.DataFrame:
    """Measure consecutive-session portfolio churn from ranking changes."""
    if candidates.empty:
        return _empty_turnover()

    c = candidates.copy()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    c = c.dropna(subset=["prediction_date", "symbol", "rank"])
    c = c[c["rank"].between(1, 20)].copy()

    sessions = sorted(c["prediction_date"].unique())
    rows = []
    for previous_date, current_date in zip(sessions, sessions[1:]):
        prev = c[c["prediction_date"] == previous_date].set_index("symbol")["rank"]
        curr = c[c["prediction_date"] == current_date].set_index("symbol")["rank"]

        row = {
            "prediction_date": pd.Timestamp(current_date),
            "previous_prediction_date": pd.Timestamp(previous_date),
        }
        for n in (5, 10, 20):
            prev_set = set(prev[prev <= n].index)
            curr_set = set(curr[curr <= n].index)
            exits = len(prev_set - curr_set)
            entries = len(curr_set - prev_set)
            row[f"top{n}_turnover_pct"] = exits / len(prev_set) * 100.0 if prev_set else np.nan
            row[f"top{n}_entries"] = entries
            row[f"top{n}_exits"] = exits
        rows.append(row)

    if not rows:
        return _empty_turnover()
    return pd.DataFrame(rows).sort_values("prediction_date").reset_index(drop=True)



COST_OUTPUT = DATA / "ranking_validation_costs.csv"
TRANSACTION_COST_BPS = (5.0, 10.0, 20.0)


def _empty_costs() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "group", "cost_bps_per_side", "rows",
        "gross_return_pct", "turnover_pct", "transaction_cost_pct",
        "net_return_pct",
    ])


def run_transaction_cost_validation(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    cost_bps_per_side: tuple[float, ...] = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Estimate daily equal-weight ranking returns after membership-change trading costs."""
    if candidates.empty or history.empty:
        return _empty_costs()

    c = candidates.copy()
    h = history.copy()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    h["symbol"] = h["symbol"].astype(str).str.strip()
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    c = c.dropna(subset=["prediction_date", "symbol", "rank"])
    c = c[c["rank"].between(1, 20)].copy()
    h = h.dropna(subset=["date", "symbol", "close"])

    sessions = sorted(c["prediction_date"].unique())
    rows = []
    previous_sets = {}
    for prediction_date in sessions:
        d = pd.Timestamp(prediction_date)
        day = c[c["prediction_date"] == d]
        future = _future_close(h, d, 1)
        base = h[h["date"].eq(d)][["symbol", "close"]].rename(columns={"close": "base_close"})
        actual = base.merge(future, on="symbol", how="inner")
        if actual.empty:
            continue
        actual["return_pct"] = (actual["future_close"] / actual["base_close"] - 1.0) * 100.0

        groups = {
            "TOP5": day[day["rank"] <= 5],
            "TOP10": day[day["rank"] <= 10],
            "TOP20": day[day["rank"] <= 20],
        }
        for group, g in groups.items():
            symbols = set(g["symbol"])
            r = actual[actual["symbol"].isin(symbols)]
            if r.empty:
                continue
            previous = previous_sets.get(group)
            turnover_pct = (
                len(previous - symbols) / len(previous) * 100.0
                if previous else 0.0
            )
            for bps in cost_bps_per_side:
                # One exit plus one entry for each replacement; cost is charged per side.
                transaction_cost_pct = (2.0 * turnover_pct / 100.0) * (float(bps) / 100.0)
                gross = r["return_pct"].mean()
                rows.append({
                    "prediction_date": d,
                    "group": group,
                    "cost_bps_per_side": float(bps),
                    "rows": len(r),
                    "gross_return_pct": gross,
                    "turnover_pct": turnover_pct,
                    "transaction_cost_pct": transaction_cost_pct,
                    "net_return_pct": gross - transaction_cost_pct,
                })
            previous_sets[group] = symbols

    if not rows:
        return _empty_costs()
    return pd.DataFrame(rows).sort_values(
        ["prediction_date", "group", "cost_bps_per_side"]
    ).reset_index(drop=True)



def _empty_sector_risk() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "prediction_date", "group", "rows", "sectors",
        "top_sector", "top_sector_weight_pct", "sector_hhi",
        "max_sector_limit_pct", "concentration_breach",
    ])


def run_sector_risk_validation(
    candidates: pd.DataFrame,
    fundamentals_history: pd.DataFrame,
    max_sector_limit_pct: float = 40.0,
) -> pd.DataFrame:
    """Measure sector concentration of ranked baskets using point-in-time snapshots."""
    if candidates.empty or fundamentals_history.empty or "sector" not in fundamentals_history.columns:
        return _empty_sector_risk()

    c = candidates.copy()
    f = fundamentals_history.copy()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    f["as_of_date"] = pd.to_datetime(f["as_of_date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
    f["symbol"] = f["symbol"].astype(str).str.upper().str.strip()
    c["rank"] = pd.to_numeric(c["rank"], errors="coerce")
    f["sector"] = f["sector"].astype(str).str.strip()
    f = f[~f["sector"].isin({"", "NAN", "NONE", "NA"})]
    c = c.dropna(subset=["prediction_date", "symbol", "rank"])
    c = c[c["rank"].between(1, 20)].copy()

    rows = []
    for d in sorted(c["prediction_date"].unique()):
        date = pd.Timestamp(d)
        eligible = f[f["as_of_date"].le(date)]
        if eligible.empty:
            continue
        snapshot_date = eligible["as_of_date"].max()
        sectors = eligible[eligible["as_of_date"].eq(snapshot_date)][["symbol", "sector"]].drop_duplicates("symbol")
        day = c[c["prediction_date"].eq(date)]
        for group, g in {
            "TOP5": day[day["rank"] <= 5],
            "TOP10": day[day["rank"] <= 10],
            "TOP20": day[day["rank"] <= 20],
        }.items():
            x = g[["symbol"]].drop_duplicates().merge(sectors, on="symbol", how="inner")
            if x.empty:
                continue
            weights = x.groupby("sector").size().div(len(x))
            top_sector = weights.idxmax()
            top_weight = float(weights.max() * 100.0)
            hhi = float((weights.pow(2)).sum() * 10000.0)
            rows.append({
                "prediction_date": date,
                "group": group,
                "rows": len(x),
                "sectors": int(weights.size),
                "top_sector": top_sector,
                "top_sector_weight_pct": top_weight,
                "sector_hhi": hhi,
                "max_sector_limit_pct": float(max_sector_limit_pct),
                "concentration_breach": bool(top_weight > max_sector_limit_pct),
            })
    if not rows:
        return _empty_sector_risk()
    return pd.DataFrame(rows).sort_values(["prediction_date", "group"]).reset_index(drop=True)


def run_ranking_diagnostics() -> pd.DataFrame:
    if not CANDIDATES.exists() or not HISTORY.exists():
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        _empty_horizons().to_csv(HORIZON_OUTPUT, index=False)
        print("Ranking diagnostics skipped: candidate history or OHLCV history missing.")
        return out

    c = pd.read_csv(CANDIDATES)
    h = pd.read_csv(HISTORY, parse_dates=["date"])
    c["target_date"] = pd.to_datetime(c["target_date"], errors="coerce").dt.normalize()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    c["symbol"] = c["symbol"].astype(str).str.strip()
    h["symbol"] = h["symbol"].astype(str).str.strip()

    for col in ["rank", "score"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["target_date", "prediction_date", "symbol", "rank", "score"])
    c = c[c["rank"].between(1, 20)].copy()
    if c.empty:
        out = _empty()
        out.to_csv(OUTPUT, index=False)
        _empty_horizons().to_csv(HORIZON_OUTPUT, index=False)
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
        _empty_horizons().to_csv(HORIZON_OUTPUT, index=False)
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
    horizons = run_multi_horizon_validation(c, h)
    horizons.to_csv(HORIZON_OUTPUT, index=False)
    mfe_mae = run_mfe_mae_validation(c, h)
    mfe_mae.to_csv(DATA / "ranking_validation_mfe_mae.csv", index=False)
    stability = run_ranking_stability(c)
    stability.to_csv(STABILITY_OUTPUT, index=False)
    turnover = run_ranking_turnover(c)
    turnover.to_csv(TURNOVER_OUTPUT, index=False)
    costs = run_transaction_cost_validation(c, h)
    costs.to_csv(COST_OUTPUT, index=False)
    if FUNDAMENTALS_HISTORY.exists():
        fh = pd.read_csv(FUNDAMENTALS_HISTORY)
        sector = run_sector_risk_validation(c, fh)
    else:
        sector = _empty_sector_risk()
    sector.to_csv(SECTOR_OUTPUT, index=False)

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
