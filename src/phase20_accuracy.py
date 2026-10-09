from __future__ import annotations

"""Phase 20: accuracy recovery diagnostics.

This is an analysis-only layer: it writes evidence reports and never changes the
production model, ranks, risk limits, or promotion decision.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVALUATIONS = DATA / "evaluations.csv"
CANDIDATES = DATA / "prediction_candidates_history.csv"
UNIVERSE = DATA / "universe.csv"
SUMMARY = DATA / "phase20_accuracy_summary.json"
SLICES = DATA / "phase20_error_slices.csv"
SYMBOLS = DATA / "phase20_symbol_stability.csv"
FEATURES = DATA / "phase20_feature_stability.csv"

FEATURE_COLUMNS = [
    "score", "technical_score", "fundamental_score", "adjusted_confidence_score",
    "confidence_score", "volatility20", "return_20d", "close_sma20_gap",
    "sector_news_score", "stock_news_score", "prediction_spread",
]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _merge_evidence(evaluations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if evaluations.empty:
        return pd.DataFrame()
    e = evaluations.copy()
    e["symbol"] = e.get("symbol", pd.Series(dtype=str)).astype(str).str.upper().str.strip()
    e["prediction_date"] = pd.to_datetime(e.get("prediction_date"), errors="coerce").dt.normalize()
    _numeric(e, ["rank", "score", "base_close", "actual_close", "predicted_close",
                 "close_abs_pct_error", "baseline_close_abs_pct_error", "close_direction_correct"])
    e["direction_correct"] = pd.to_numeric(e.get("close_direction_correct"), errors="coerce")
    e["close_error_pct"] = pd.to_numeric(e.get("close_abs_pct_error"), errors="coerce") * 100
    e["baseline_close_error_pct"] = pd.to_numeric(e.get("baseline_close_abs_pct_error"), errors="coerce") * 100
    if candidates.empty or not {"prediction_date", "symbol"}.issubset(candidates.columns):
        return e
    c = candidates.copy()
    c["symbol"] = c["symbol"].astype(str).str.upper().str.strip()
    c["prediction_date"] = pd.to_datetime(c["prediction_date"], errors="coerce").dt.normalize()
    cols = ["prediction_date", "symbol"] + [x for x in FEATURE_COLUMNS + [
        "confidence_tier", "risk_level", "market_regime", "index_market_regime",
        "sector", "sector_name", "selection_method", "selected"
    ] if x in c.columns]
    c = c[cols].drop_duplicates(["prediction_date", "symbol"], keep="last")
    overlap = [x for x in c.columns if x not in {"prediction_date", "symbol"} and x in e.columns]
    e = e.drop(columns=overlap, errors="ignore")
    return e.merge(c, on=["prediction_date", "symbol"], how="left")


def _slice_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dimension", "bucket", "rows", "sessions", "direction_accuracy_pct", "close_mape_pct", "baseline_close_mape_pct", "mape_improvement_pct"])
    x = df.copy()
    specs: list[tuple[str, pd.Series]] = []
    if "rank" in x:
        rank = pd.to_numeric(x["rank"], errors="coerce")
        specs.extend([("rank_bucket", pd.Series(np.select([rank.le(5), rank.le(10)], ["TOP5", "TOP10"], default="RANK_11_PLUS"), index=x.index))])
    for col, label in [
        ("confidence_tier", "confidence_tier"), ("risk_level", "risk_level"),
        ("market_regime", "market_regime"), ("index_market_regime", "index_market_regime"),
        ("selection_method", "selection_method"), ("sector", "sector"), ("sector_name", "sector"),
    ]:
        if col in x and x[col].notna().any():
            specs.append((label, x[col].fillna("UNKNOWN").astype(str).str.upper()))
    if "stock_news_score" in x:
        v = pd.to_numeric(x["stock_news_score"], errors="coerce")
        specs.append(("stock_news", pd.Series(np.select([v.gt(0.05), v.lt(-0.05)], ["POSITIVE", "NEGATIVE"], default="NEUTRAL_OR_MISSING"), index=x.index)))
    if "sector_news_score" in x:
        v = pd.to_numeric(x["sector_news_score"], errors="coerce")
        specs.append(("sector_news", pd.Series(np.select([v.gt(0.05), v.lt(-0.05)], ["POSITIVE", "NEGATIVE"], default="NEUTRAL_OR_MISSING"), index=x.index)))
    if "volatility20" in x:
        v = pd.to_numeric(x["volatility20"], errors="coerce")
        try:
            specs.append(("volatility", pd.qcut(v, q=3, labels=["LOW", "MEDIUM", "HIGH"], duplicates="drop").astype(str)))
        except ValueError:
            pass
    if "prediction_spread" in x:
        v = pd.to_numeric(x["prediction_spread"], errors="coerce")
        try:
            specs.append(("prediction_spread", pd.qcut(v, q=3, labels=["LOW", "MEDIUM", "HIGH"], duplicates="drop").astype(str)))
        except ValueError:
            pass
    rows = []
    for dimension, buckets in specs:
        temp = x.assign(_bucket=buckets)
        for bucket, g in temp.groupby("_bucket", dropna=False):
            correct = pd.to_numeric(g["direction_correct"], errors="coerce").dropna()
            mape = pd.to_numeric(g["close_error_pct"], errors="coerce").dropna()
            base = pd.to_numeric(g["baseline_close_error_pct"], errors="coerce").dropna()
            model_mape = float(mape.mean()) if len(mape) else np.nan
            base_mape = float(base.mean()) if len(base) else np.nan
            rows.append({
                "dimension": dimension, "bucket": str(bucket), "rows": int(len(g)),
                "sessions": int(g["prediction_date"].nunique()) if "prediction_date" in g else 0,
                "direction_accuracy_pct": float(correct.mean() * 100) if len(correct) else np.nan,
                "close_mape_pct": model_mape, "baseline_close_mape_pct": base_mape,
                "mape_improvement_pct": float((base_mape-model_mape)/base_mape*100) if np.isfinite(base_mape) and base_mape > 0 and np.isfinite(model_mape) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["dimension", "rows"], ascending=[True, False]) if rows else pd.DataFrame()


def _symbol_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "sessions", "direction_accuracy_pct", "close_mape_pct", "baseline_close_mape_pct", "mape_improvement_pct", "instability_flag", "diagnostic"]
    if df.empty or "symbol" not in df:
        return pd.DataFrame(columns=cols)
    rows = []
    for symbol, g in df.groupby("symbol"):
        direction = pd.to_numeric(g["direction_correct"], errors="coerce").dropna()
        mape = pd.to_numeric(g["close_error_pct"], errors="coerce").dropna()
        base = pd.to_numeric(g["baseline_close_error_pct"], errors="coerce").dropna()
        acc = float(direction.mean()*100) if len(direction) else np.nan
        model = float(mape.mean()) if len(mape) else np.nan
        baseline = float(base.mean()) if len(base) else np.nan
        lift = float((baseline-model)/baseline*100) if np.isfinite(baseline) and baseline > 0 and np.isfinite(model) else np.nan
        enough = len(g) >= 3
        flag = bool(enough and ((np.isfinite(acc) and acc < 40) or (np.isfinite(lift) and lift < -5)))
        reasons = []
        if enough and np.isfinite(acc) and acc < 40: reasons.append("direction_accuracy_below_40pct")
        if enough and np.isfinite(lift) and lift < -5: reasons.append("close_mape_over_5pct_worse_than_baseline")
        rows.append({"symbol": symbol, "sessions": int(g["prediction_date"].nunique()),
                     "direction_accuracy_pct": acc, "close_mape_pct": model,
                     "baseline_close_mape_pct": baseline, "mape_improvement_pct": lift,
                     "instability_flag": flag, "diagnostic": ";".join(reasons) if reasons else "insufficient_or_no_threshold_breach"})
    return pd.DataFrame(rows, columns=cols).sort_values(["instability_flag", "sessions"], ascending=[False, False])


def _feature_report(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["feature", "rows", "folds", "median_correlation", "sign_consistency_pct", "stable_candidate", "status"]
    if df.empty or "direction_correct" not in df:
        return pd.DataFrame(columns=columns)
    x = df.copy()
    x["direction_correct"] = pd.to_numeric(x["direction_correct"], errors="coerce")
    dates = sorted(pd.to_datetime(x.get("prediction_date"), errors="coerce").dropna().unique())
    if len(dates) < 5:
        return pd.DataFrame([{"feature": f, "rows": int(len(x)), "folds": 0, "median_correlation": np.nan,
                              "sign_consistency_pct": np.nan, "stable_candidate": False,
                              "status": "COLLECTING_MINIMUM_5_SESSIONS"} for f in FEATURE_COLUMNS if f in x.columns], columns=columns)
    # Chronological, non-overlapping folds; never shuffle future observations into earlier folds.
    fold_dates = [list(a) for a in np.array_split(np.array(dates, dtype=object), min(5, len(dates))) if len(a)]
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature not in x:
            continue
        correlations = []
        for fd in fold_dates:
            g = x[x["prediction_date"].isin(fd)]
            a = pd.to_numeric(g[feature], errors="coerce")
            b = g["direction_correct"]
            valid = pd.DataFrame({"a": a, "b": b}).dropna()
            if len(valid) >= 30 and valid["a"].nunique() > 1 and valid["b"].nunique() > 1:
                corr = valid["a"].corr(valid["b"], method="spearman")
                if pd.notna(corr): correlations.append(float(corr))
        consistent = (max(sum(c > 0 for c in correlations), sum(c < 0 for c in correlations))/len(correlations)*100) if correlations else 0.0
        median_corr = float(np.median(correlations)) if correlations else np.nan
        stable = bool(len(correlations) >= 5 and abs(median_corr) >= 0.03 and consistent >= 70)
        rows.append({"feature": feature, "rows": int(pd.to_numeric(x[feature], errors="coerce").notna().sum()),
                     "folds": len(correlations), "median_correlation": median_corr,
                     "sign_consistency_pct": consistent, "stable_candidate": stable,
                     "status": "STABLE_CANDIDATE_REQUIRES_REVIEW" if stable else "NOT_STABLE_OR_INSUFFICIENT_EVIDENCE"})
    return pd.DataFrame(rows, columns=columns)


def run_phase20() -> dict:
    evaluations = _read(EVALUATIONS)
    candidates = _read(CANDIDATES)
    df = _merge_evidence(evaluations, candidates)
    slices = _slice_report(df)
    symbols = _symbol_report(df)
    features = _feature_report(df)
    slices.to_csv(SLICES, index=False)
    symbols.to_csv(SYMBOLS, index=False)
    features.to_csv(FEATURES, index=False)
    direction = pd.to_numeric(df.get("direction_correct", pd.Series(dtype=float)), errors="coerce").dropna()
    mape = pd.to_numeric(df.get("close_error_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    base = pd.to_numeric(df.get("baseline_close_error_pct", pd.Series(dtype=float)), errors="coerce").dropna()
    summary = {
        "phase": 20,
        "as_of": str(pd.to_datetime(df.get("target_date", pd.Series(dtype=str)), errors="coerce").max().date()) if not df.empty and pd.to_datetime(df.get("target_date"), errors="coerce").notna().any() else "",
        "status": "PASS" if not df.empty else "COLLECTING",
        "analysis_only": True,
        "production_model_changed": False,
        "evaluation_rows": int(len(df)),
        "evaluation_sessions": int(df["prediction_date"].nunique()) if "prediction_date" in df else 0,
        "direction_accuracy_pct": float(direction.mean()*100) if len(direction) else None,
        "close_mape_pct": float(mape.mean()) if len(mape) else None,
        "baseline_close_mape_pct": float(base.mean()) if len(base) else None,
        "unstable_symbols_flagged": int(symbols["instability_flag"].sum()) if not symbols.empty else 0,
        "stable_feature_candidates": int(features["stable_candidate"].sum()) if not features.empty else 0,
        "feature_evidence_status": "REVIEW_REQUIRED" if not features.empty and features["stable_candidate"].any() else "NO_STABLE_FEATURES_IDENTIFIED",
        "reports": {"error_slices": str(SLICES.relative_to(ROOT)), "symbol_stability": str(SYMBOLS.relative_to(ROOT)), "feature_stability": str(FEATURES.relative_to(ROOT))},
        "guardrail": "No feature selection, production ranking, risk setting, or promotion state is changed by Phase 20.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    run_phase20()
