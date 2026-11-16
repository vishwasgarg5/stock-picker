from __future__ import annotations
"""Leakage-safe, analysis-only feature-signal audit for next-session returns."""
import json
import numpy as np
import pandas as pd
from .pipeline import DATA, FEATURE_COLUMNS, HISTORY_FILE, add_targets, features

OUTPUT_JSON = DATA / "phase124_feature_signal_summary.json"
OUTPUT_CSV = DATA / "phase124_feature_signal_by_fold.csv"

def _spearman(frame: pd.DataFrame, feature: str, target: str) -> float:
    pair = frame[[feature, target]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 10 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
        return float("nan")
    return float(pair[feature].corr(pair[target], method="spearman"))

def analyse_feature_signal(frame: pd.DataFrame, feature_columns: list[str] | None = None,
                           n_folds: int = 4, min_cross_section: int = 10) -> tuple[dict, pd.DataFrame]:
    """Expanding folds; training labels must end before each test feature date."""
    cols = feature_columns or FEATURE_COLUMNS
    required = {"date", "target_date", "symbol", "target_close_return", *cols}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Phase 124 input missing columns: {missing}")
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["target_close_return"] = pd.to_numeric(x["target_close_return"], errors="coerce")
    x = x.dropna(subset=["date", "target_date", "symbol", "target_close_return"])
    x = x.loc[x["target_date"] > x["date"]].copy()
    if x.duplicated(["date", "symbol"]).any():
        raise ValueError("Phase 124 duplicate date/symbol feature keys")
    dates = np.array(sorted(x["date"].unique()))
    initial = max(2, int(len(dates) * 0.60))
    if len(dates) - initial < n_folds:
        raise ValueError("Insufficient dates for requested chronological folds")
    blocks = [b for b in np.array_split(dates[initial:], n_folds) if len(b)]
    records = []
    for fold, block in enumerate(blocks, start=1):
        test_start, test_end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        train = x.loc[x["target_date"] < test_start].copy()
        test = x.loc[x["date"].isin(block)].copy()
        if train.empty or test.empty:
            continue
        for feature in cols:
            train_corr = _spearman(train, feature, "target_close_return")
            ics = []
            for _, session in test.groupby("date", sort=True):
                if session["symbol"].nunique() < min_cross_section:
                    continue
                value = _spearman(session, feature, "target_close_return")
                if np.isfinite(value):
                    ics.append(value)
            records.append({
                "fold": fold, "train_end_target_date": train["target_date"].max().date().isoformat(),
                "test_start_date": test_start.date().isoformat(), "test_end_date": test_end.date().isoformat(),
                "feature": feature, "train_spearman": train_corr, "test_session_count": len(ics),
                "test_mean_rank_ic": float(np.mean(ics)) if ics else None,
                "test_std_rank_ic": float(np.std(ics, ddof=1)) if len(ics) > 1 else None,
                "test_positive_session_fraction": float(np.mean(np.asarray(ics) > 0)) if ics else None,
                "train_rows": int(len(train)), "test_rows": int(len(test)),
                "strict_label_cutoff": bool(train["target_date"].max() < test_start),
            })
    fold_frame = pd.DataFrame(records)
    if fold_frame.empty:
        raise ValueError("Phase 124 produced no valid fold metrics")
    summaries = []
    for feature, group in fold_frame.groupby("feature", sort=True):
        values = pd.to_numeric(group["test_mean_rank_ic"], errors="coerce").dropna()
        summaries.append({
            "feature": feature, "valid_folds": int(values.size),
            "mean_oos_rank_ic": float(values.mean()) if len(values) else None,
            "std_oos_rank_ic": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sign_consistent_folds": int((np.sign(values) == np.sign(values.mean())).sum()) if len(values) else 0,
            "mean_train_spearman": float(pd.to_numeric(group["train_spearman"], errors="coerce").mean()),
        })
    summary = {
        "phase": 124, "status": "ANALYSIS_COMPLETE",
        "purpose": "Research-only next-session feature signal audit; no model fitting or production changes",
        "rows": int(len(x)), "symbols": int(x["symbol"].nunique()),
        "feature_dates": int(x["date"].nunique()), "target_sessions": int(x["target_date"].nunique()),
        "folds_requested": int(n_folds), "folds_completed": int(fold_frame["fold"].nunique()),
        "strict_no_lookahead": bool(fold_frame["strict_label_cutoff"].all()),
        "method": "Expanding chronological folds; train labels strictly precede each test block; test signal is per-session cross-sectional Spearman rank IC",
        "feature_summary": summaries,
        "production": {"champion": "V1", "changed": False, "challengers_shadow_only": True, "automatic_promotion": False},
    }
    return summary, fold_frame

def run() -> dict:
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise RuntimeError(f"Phase 124 history missing columns: {missing}")
    labelled = add_targets(features(history))
    labelled["target_date"] = labelled.groupby("symbol", sort=False)["date"].shift(-1)
    summary, fold_frame = analyse_feature_signal(labelled, FEATURE_COLUMNS, n_folds=4)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fold_frame.to_csv(OUTPUT_CSV, index=False)
    compact = [{"feature": r["feature"], "mean_oos_rank_ic": r["mean_oos_rank_ic"],
                "sign_consistent_folds": r["sign_consistent_folds"]} for r in summary["feature_summary"]]
    print(json.dumps({"phase": 124, "rows": summary["rows"], "symbols": summary["symbols"],
        "feature_dates": summary["feature_dates"], "folds_completed": summary["folds_completed"],
        "strict_no_lookahead": summary["strict_no_lookahead"],
        "features_by_abs_oos_rank_ic": sorted([r for r in compact if r["mean_oos_rank_ic"] is not None],
            key=lambda r: abs(r["mean_oos_rank_ic"]), reverse=True)[:8],
        "outputs": [str(OUTPUT_JSON), str(OUTPUT_CSV)]}, indent=2))
    return summary

if __name__ == "__main__":
    run()
