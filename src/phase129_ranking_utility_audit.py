from __future__ import annotations
"""Phase 129: cross-sectional ranking utility audit; diagnostic only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES_FILE = DATA / "prediction_candidates_history.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_JSON = DATA / "phase129_ranking_utility_summary.json"
OUTPUT_CSV = DATA / "phase129_ranking_utility_by_session.csv"
COST_BPS_PER_SIDE = 10.0
MIN_REVIEW_SESSIONS = 17


def _prepare(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    required_c = {"prediction_date", "target_date", "symbol", "rank", "score", "base_close"}
    required_h = {"date", "symbol", "close"}
    if missing := sorted(required_c.difference(candidates.columns)):
        raise ValueError(f"Phase 129 candidates missing columns: {missing}")
    if missing := sorted(required_h.difference(history.columns)):
        raise ValueError(f"Phase 129 history missing columns: {missing}")
    c = candidates.copy()
    h = history.copy()
    c["prediction_date"] = pd.to_datetime(c.prediction_date, errors="coerce").dt.normalize()
    c["target_date"] = pd.to_datetime(c.target_date, errors="coerce").dt.normalize()
    h["date"] = pd.to_datetime(h.date, errors="coerce").dt.normalize()
    c["symbol"] = c.symbol.astype(str).str.upper().str.strip()
    h["symbol"] = h.symbol.astype(str).str.upper().str.strip()
    for col in ("rank", "score", "base_close"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    h["close"] = pd.to_numeric(h["close"], errors="coerce")
    c = c.dropna(subset=["prediction_date", "target_date", "symbol", "rank", "score", "base_close"])
    h = h.dropna(subset=["date", "symbol", "close"])
    c = c.loc[(c.target_date > c.prediction_date) & c["rank"].between(1, 20) & (c.base_close > 0)].copy()
    if c.duplicated(["prediction_date", "symbol"]).any():
        raise ValueError("Phase 129 duplicate prediction_date/symbol keys")
    if h.duplicated(["date", "symbol"]).any():
        h = h.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    actual = h.rename(columns={"date": "target_date", "close": "actual_close"})
    x = c.merge(actual[["target_date", "symbol", "actual_close"]], on=["target_date", "symbol"], how="left", validate="many_to_one")
    x = x.loc[np.isfinite(x["actual_close"]) & (x["actual_close"] > 0)].copy()
    expected = c.groupby("prediction_date").size()
    matched = x.groupby("prediction_date").size()
    complete_dates = expected.index[expected.eq(matched.reindex(expected.index, fill_value=0))]
    x = x.loc[x.prediction_date.isin(complete_dates)].copy()
    if x.empty:
        raise ValueError("Phase 129 has no complete sessions with exact target-date actuals")
    x["actual_return_pct"] = (x.actual_close / x.base_close - 1.0) * 100.0
    return x


def _market_volatility(history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    h["date"] = pd.to_datetime(h.date, errors="coerce").dt.normalize()
    h["symbol"] = h.symbol.astype(str).str.upper().str.strip()
    h["close"] = pd.to_numeric(h.close, errors="coerce")
    h = h.dropna(subset=["date", "symbol", "close"]).sort_values(["date", "symbol"])
    h = h.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
    h["daily_return"] = h.groupby("symbol").close.pct_change()
    h["vol20"] = h.groupby("symbol").daily_return.transform(lambda s: s.rolling(20, min_periods=15).std())
    return h.groupby("date", as_index=False).vol20.median().rename(columns={"vol20": "market_vol20"})


def evaluate_ranking_utility(candidates: pd.DataFrame, history: pd.DataFrame,
                             n_folds: int = 4, min_review_sessions: int = MIN_REVIEW_SESSIONS,
                             cost_bps_per_side: float = COST_BPS_PER_SIDE) -> tuple[dict, pd.DataFrame]:
    if n_folds < 2 or min_review_sessions < 1 or cost_bps_per_side < 0:
        raise ValueError("Invalid Phase 129 evaluation configuration")
    original_sessions = pd.to_datetime(candidates["prediction_date"], errors="coerce").dt.normalize().nunique()
    x = _prepare(candidates, history)
    excluded_incomplete_sessions = int(original_sessions - x.prediction_date.nunique())
    dates = np.array(sorted(x.prediction_date.unique()))
    initial = min(len(dates) - 1, max(2, int(len(dates) * 0.60)))
    available_test_sessions = len(dates) - initial
    if available_test_sessions < 1:
        raise ValueError("At least three complete prediction sessions are required for chronological evaluation")
    effective_folds = min(n_folds, available_test_sessions)
    blocks = [b for b in np.array_split(dates[initial:], effective_folds) if len(b)]
    vol = _market_volatility(history)
    x = x.merge(vol, left_on="prediction_date", right_on="date", how="left").drop(columns=["date"])
    x["market_vol20"] = pd.to_numeric(x["market_vol20"], errors="coerce")
    rows = []
    previous_top5: set[str] | None = None
    for fold, block in enumerate(blocks, 1):
        start = pd.Timestamp(block[0])
        prior_vol = x.loc[x.prediction_date < start, ["prediction_date", "market_vol20"]].drop_duplicates()
        prior_vol = prior_vol.market_vol20.dropna()
        if len(prior_vol) >= 5:
            low_cut, high_cut = prior_vol.quantile([1/3, 2/3]).tolist()
        else:
            low_cut = high_cut = np.nan
        for d in block:
            day = x.loc[x.prediction_date.eq(pd.Timestamp(d))].sort_values(["rank", "symbol"])
            if day.symbol.nunique() < 10:
                continue
            score_ic = day[["score", "actual_return_pct"]].corr(method="spearman").iloc[0, 1]
            top = day.loc[day["rank"].between(1, 5)]
            max_rank = float(day["rank"].max())
            bottom = day.loc[day["rank"].gt(max_rank - 5) & day["rank"].le(max_rank)]
            if len(top) < 5 or len(bottom) < 5 or set(top.symbol).intersection(bottom.symbol):
                continue
            top_symbols = set(top.symbol)
            if previous_top5 is None:
                entries, exits = len(top_symbols), 0
            else:
                entries, exits = len(top_symbols - previous_top5), len(previous_top5 - top_symbols)
            turnover_pct = (entries + exits) / 5.0 * 100.0
            cost_pct = (entries + exits) / 5.0 * cost_bps_per_side / 100.0
            top_return = float(top.actual_return_pct.mean())
            bottom_return = float(bottom.actual_return_pct.mean())
            market_vol = float(day.market_vol20.median()) if day.market_vol20.notna().any() else np.nan
            if not np.isfinite(market_vol) or not np.isfinite(low_cut):
                regime = "UNKNOWN"
            elif market_vol <= low_cut:
                regime = "LOW_VOL"
            elif market_vol <= high_cut:
                regime = "MID_VOL"
            else:
                regime = "HIGH_VOL"
            rows.append({
                "fold": fold, "prediction_date": pd.Timestamp(d).date().isoformat(),
                "rank_ic_spearman": float(score_ic) if np.isfinite(score_ic) else np.nan,
                "top5_mean_return_pct": top_return, "bottom5_mean_return_pct": bottom_return,
                "top_bottom_spread_pct": top_return - bottom_return,
                "top5_hit_rate_pct": float((top.actual_return_pct > 0).mean() * 100),
                "top5_turnover_pct": turnover_pct, "transaction_cost_pct": cost_pct,
                "top5_net_return_pct": top_return - cost_pct,
                "market_vol20": market_vol, "volatility_regime": regime,
                "regime_cutoffs_train_only": bool(np.isfinite(low_cut) and np.isfinite(high_cut)),
            })
            previous_top5 = top_symbols
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("Phase 129 produced no complete Top-5 versus bottom-5 sessions")
    regime_summary = []
    for regime, group in detail.groupby("volatility_regime", sort=True):
        regime_summary.append({
            "regime": regime, "sessions": int(len(group)),
            "mean_rank_ic": float(group.rank_ic_spearman.mean()),
            "mean_top_bottom_spread_pct": float(group.top_bottom_spread_pct.mean()),
            "mean_top5_net_return_pct": float(group.top5_net_return_pct.mean()),
        })
    summary = {
        "phase": 129, "status": "READY_FOR_REVIEW" if len(detail) >= min_review_sessions else "INSUFFICIENT_SESSIONS",
        "candidate_rows_evaluated": int(len(x)), "excluded_incomplete_sessions": excluded_incomplete_sessions, "symbols": int(x.symbol.nunique()),
        "prediction_sessions_evaluated": int(detail.prediction_date.nunique()),
        "folds_completed": int(detail.fold.nunique()), "requested_folds": int(n_folds), "minimum_review_sessions": int(min_review_sessions),
        "rank_ic_spearman_mean": float(detail.rank_ic_spearman.mean()),
        "rank_ic_positive_session_fraction": float((detail.rank_ic_spearman > 0).mean()),
        "top_bottom_spread_mean_pct": float(detail.top_bottom_spread_pct.mean()),
        "top5_gross_mean_return_pct": float(detail.top5_mean_return_pct.mean()),
        "top5_net_mean_return_pct": float(detail.top5_net_return_pct.mean()),
        "top5_mean_turnover_pct": float(detail.top5_turnover_pct.mean()),
        "cost_bps_per_side": float(cost_bps_per_side),
        "volatility_regimes": regime_summary,
        "strict_chronological_regime_cutoffs": bool(detail.regime_cutoffs_train_only.all()),
        "method": "Exact target-date close-to-close returns; score/return cross-sectional Spearman IC; rank 1-5 versus rank 16-20 spread; Top-5 membership turnover and explicit per-side cost; expanding chronological test blocks; volatility regime cutoffs estimated only from dates before each test block.",
        "performance_gate_met": False,
        "production": {"champion": "V1", "challengers": "SHADOW_ONLY", "changed": False, "automatic_promotion": False},
        "note": "17 sessions is a minimum evidence-review threshold only; it is not a performance or promotion pass."
    }
    return summary, detail


def run() -> dict:
    candidates = pd.read_csv(CANDIDATES_FILE)
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    summary, detail = evaluate_ranking_utility(candidates, history)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    detail.to_csv(OUTPUT_CSV, index=False)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
