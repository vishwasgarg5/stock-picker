from __future__ import annotations
"""Phase 125: chronological feature ablation; research-only, never promotes models."""
import json
import numpy as np
import pandas as pd
from .pipeline import DATA, FEATURE_COLUMNS, HISTORY_FILE, add_targets, features, _fit_ensemble, _ensemble_predict

OUTPUT_JSON = DATA / "phase125_feature_ablation_summary.json"
OUTPUT_CSV = DATA / "phase125_feature_ablation_by_fold.csv"
ABLATIONS = {
    "incumbent_all_features": list(FEATURE_COLUMNS),
    "without_price_level_indicators": [c for c in FEATURE_COLUMNS if c not in {"sma20", "sma50", "ema20", "ema50", "macd", "macd_signal"}],
    "compact_returns_risk_volume": ["return_1d", "return_5d", "return_20d", "rsi14", "volume_ratio", "atr14_pct", "bb_position", "range_pct", "volatility20", "volume_trend5"],
}

def _price_mape(predicted_return: np.ndarray, actual_return: np.ndarray) -> float:
    valid = np.isfinite(predicted_return) & np.isfinite(actual_return) & (actual_return > -1.0)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(predicted_return[valid] - actual_return[valid]) / (1.0 + actual_return[valid])) * 100.0)

def run_ablation(frame: pd.DataFrame, n_folds: int = 4) -> tuple[dict, pd.DataFrame]:
    required = {"date", "target_date", "symbol", "target_close_return", *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Phase 125 input missing columns: {missing}")
    if n_folds < 2:
        raise ValueError("Phase 125 requires at least two chronological folds")
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    for col in FEATURE_COLUMNS + ["target_close_return"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["date", "target_date", "symbol", "target_close_return"])
    x = x.loc[(x["target_date"] > x["date"]) & (x["target_close_return"] > -1.0)].copy()
    if x.duplicated(["date", "symbol"]).any():
        raise ValueError("Phase 125 duplicate date/symbol keys")
    dates = np.array(sorted(x["date"].unique()))
    initial = max(2, int(len(dates) * 0.60))
    if len(dates) - initial < n_folds:
        raise ValueError("Insufficient dates for requested folds")
    blocks = [b for b in np.array_split(dates[initial:], n_folds) if len(b)]
    rows = []
    for fold, block in enumerate(blocks, 1):
        test_start = pd.Timestamp(block[0])
        train = x.loc[x["target_date"] < test_start].copy()
        test = x.loc[x["date"].isin(block)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Phase 125 fold {fold} has empty train or test partition")
        # Lock one common complete-case cohort before comparing feature sets.
        # Otherwise an ablation could win merely by dropping difficult rows.
        common_train = train.dropna(subset=FEATURE_COLUMNS).copy()
        common_test = test.dropna(subset=FEATURE_COLUMNS).copy()
        if len(common_train) < 500 or len(common_test) < 10:
            raise ValueError(f"Phase 125 insufficient common complete rows in fold {fold}")
        y_test = common_test["target_close_return"].to_numpy(dtype=float)
        baseline_mape = _price_mape(np.zeros(len(common_test)), y_test)
        for name, columns in ABLATIONS.items():
            bundle = _fit_ensemble(common_train[columns], common_train["target_close_return"])
            predicted, _ = _ensemble_predict(bundle, common_test[columns])
            actual = common_test["target_close_return"].to_numpy(dtype=float)
            pred_mape = _price_mape(predicted, actual)
            rows.append({
                "fold": fold, "variant": name,
                "train_end_target_date": train["target_date"].max().date().isoformat(),
                "test_start_date": test["date"].min().date().isoformat(),
                "test_end_date": test["date"].max().date().isoformat(),
                "train_rows": int(len(common_train)), "test_rows": int(len(common_test)),
                "baseline_mape_pct": baseline_mape,
                "model_mape_pct": pred_mape,
                "delta_vs_baseline_pp": pred_mape - baseline_mape,
                "directional_accuracy_pct": float(np.mean(np.sign(predicted) == np.sign(actual)) * 100.0),
                "signed_bias_pp": float(np.mean(predicted - actual) * 100.0),
                "mean_abs_predicted_move_pct": float(np.mean(np.abs(predicted)) * 100.0),
                "mean_abs_realized_move_pct": float(np.mean(np.abs(actual)) * 100.0),
                "strict_label_cutoff": bool(train["target_date"].max() < test_start),
            })
    result = pd.DataFrame(rows)
    summaries = []
    for name, group in result.groupby("variant", sort=False):
        deltas = group["delta_vs_baseline_pp"].to_numpy(dtype=float)
        summaries.append({
            "variant": name, "features": ABLATIONS[name],
            "mean_fold_mape_pct": float(group["model_mape_pct"].mean()),
            "mean_baseline_mape_pct": float(group["baseline_mape_pct"].mean()),
            "mean_delta_vs_baseline_pp": float(deltas.mean()),
            "folds_beating_baseline": int(np.sum(deltas < 0)),
            "folds_tested": int(len(group)),
            "mean_directional_accuracy_pct": float(group["directional_accuracy_pct"].mean()),
            "mean_signed_bias_pp": float(group["signed_bias_pp"].mean()),
            "mean_abs_predicted_move_pct": float(group["mean_abs_predicted_move_pct"].mean()),
            "mean_abs_realized_move_pct": float(group["mean_abs_realized_move_pct"].mean()),
        })
    winner = min(summaries, key=lambda item: item["mean_delta_vs_baseline_pp"])
    qualified = winner["mean_delta_vs_baseline_pp"] < 0 and winner["folds_beating_baseline"] >= max(3, int(np.ceil(n_folds * 0.75)))
    summary = {
        "phase": 125, "status": "ABLATION_COMPLETE",
        "rows": int(len(x)), "symbols": int(x["symbol"].nunique()),
        "feature_dates": int(x["date"].nunique()), "folds_completed": int(result["fold"].nunique()),
        "strict_no_lookahead": bool(result["strict_label_cutoff"].all()),
        "method": "Four expanding chronological folds; training labels strictly precede test block; same incumbent 70/30 HistGradientBoosting/ExtraTrees ensemble per predeclared feature set; zero-return baseline; no test-fold feature selection.",
        "variants": summaries, "best_variant": winner["variant"],
        "performance_gate_met": bool(qualified),
        "decision": "RESEARCH_CANDIDATE_ONLY" if qualified else "NO_FEATURE_ABLATION_QUALIFIED",
        "production": {"champion": "V1", "changed": False, "automatic_promotion": False},
    }
    return summary, result

def run() -> dict:
    history = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    labelled = add_targets(features(history))
    labelled["target_date"] = labelled.groupby("symbol", sort=False)["date"].shift(-1)
    summary, detail = run_ablation(labelled, n_folds=4)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    detail.to_csv(OUTPUT_CSV, index=False)
    print(json.dumps({"phase": 125, "rows": summary["rows"], "symbols": summary["symbols"],
        "folds_completed": summary["folds_completed"], "strict_no_lookahead": summary["strict_no_lookahead"],
        "best_variant": summary["best_variant"], "performance_gate_met": summary["performance_gate_met"],
        "variants": summary["variants"], "outputs": [str(OUTPUT_JSON), str(OUTPUT_CSV)]}, indent=2))
    return summary

if __name__ == "__main__":
    run()
