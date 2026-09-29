from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from src.pipeline import FEATURE_COLUMNS, TARGETS, features, add_targets, rank_stocks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY_FILE = DATA / "ohlcv.csv"
OUTPUT_FILE = DATA / "walk_forward_evaluations.csv"
SUMMARY_FILE = DATA / "walk_forward_summary.csv"
SELECTION_FILE = DATA / "model_selection.csv"
MIN_SELECTION_SESSIONS = 12
MIN_RELATIVE_IMPROVEMENT = 0.01
MIN_REGIME_RELATIVE_IMPROVEMENT = 0.01

MIN_TRAIN_ROWS = 500
LOOKBACK_MONTHS = 24
CHECKPOINTS = 24


def _fit_models(train_rows: pd.DataFrame) -> dict[str, dict]:
    models = {}
    x = train_rows[FEATURE_COLUMNS]
    for name, target in TARGETS.items():
        hist = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=42,
        )
        extra = ExtraTreesRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=4,
            max_features=0.8, n_jobs=-1, random_state=42,
        )
        y = train_rows[target]
        hist.fit(x, y)
        extra.fit(x, y)
        models[name] = {"models": [hist, extra], "weights": [0.70, 0.30]}
    return models


def _predict(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    models = bundle["models"]
    weights = np.asarray(bundle["weights"], dtype=float)
    weights = weights / weights.sum()
    preds = np.column_stack([m.predict(x) for m in models])
    return preds @ weights

def _safe_relative_improvement(baseline: float, model: float, min_baseline: float = 1e-3) -> float:
    """Return relative improvement only when the baseline is numerically meaningful."""
    baseline = float(baseline)
    model = float(model)
    if not np.isfinite(baseline) or not np.isfinite(model) or baseline < min_baseline:
        return np.nan
    return (baseline - model) / baseline * 100.0


def _fit_challenger(x: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=42,
    )
    model.fit(x, y)
    return model

def _select_model(output: pd.DataFrame) -> pd.DataFrame:
    sessions = int(output["target_date"].nunique())
    ensemble_mape = float(output["close_abs_pct_error"].mean())
    challenger_mape = float(output["challenger_close_abs_pct_error"].mean())
    improvement = (challenger_mape - ensemble_mape) / max(challenger_mape, 1e-12)
    eligible = sessions >= MIN_SELECTION_SESSIONS
    passed = bool(eligible and improvement >= MIN_RELATIVE_IMPROVEMENT)
    return pd.DataFrame([{
        "as_of": output["target_date"].max(),
        "sessions": sessions,
        "ensemble_close_mape_pct": ensemble_mape * 100,
        "challenger_close_mape_pct": challenger_mape * 100,
        "ensemble_relative_improvement_vs_challenger_pct": improvement,
        "minimum_sessions_required": MIN_SELECTION_SESSIONS,
        "minimum_relative_improvement_pct": MIN_RELATIVE_IMPROVEMENT * 100,
        "selected_model": "ensemble_v1",
        "promotion_gate_passed": passed,
        "decision": "Ensemble passed promotion gate." if passed else "Ensemble retained; promotion gate not yet passed."
    }])


def _checkpoints(dates: pd.Series) -> list[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates, errors="coerce").dropna().dt.normalize().unique())
    unique = unique.sort_values()
    if len(unique) < 2:
        return []
    start = unique.max() - pd.DateOffset(months=LOOKBACK_MONTHS)
    eligible = unique[unique >= start]
    idx = np.linspace(0, len(eligible) - 2, min(CHECKPOINTS, max(1, len(eligible) - 1)), dtype=int)
    return list(eligible.iloc[np.unique(idx)])


def _historical_production_selection(feat: pd.DataFrame, checkpoint: pd.Timestamp) -> pd.DataFrame | None:
    if CANDIDATE_HISTORY_FILE.exists():
        h = pd.read_csv(CANDIDATE_HISTORY_FILE)
        h["prediction_date"] = pd.to_datetime(h["prediction_date"], errors="coerce").dt.normalize()
        selected = pd.to_numeric(h.get("selected", 0), errors="coerce").fillna(0)
        rows = h[(h["prediction_date"] == pd.Timestamp(checkpoint).normalize()) & (selected == 1)].copy()
        if len(rows) >= 10:
            rows = rows.sort_values(["rank", "symbol"]).head(10)
            rows["market_regime"] = "RECORDED"
            return rows[["symbol", "rank", "score", "market_regime"]]
    if not FUNDAMENTALS_HISTORY_FILE.exists():
        return None
    h = pd.read_csv(FUNDAMENTALS_HISTORY_FILE)
    h["as_of_date"] = pd.to_datetime(h["as_of_date"], errors="coerce").dt.normalize()
    cp = pd.Timestamp(checkpoint).normalize()
    h = h[(h["as_of_date"] <= cp) & (h["as_of_date"] >= cp - pd.Timedelta(days=7))]
    if h.empty:
        return None
    snap = h[h["as_of_date"] == h["as_of_date"].max()].drop(columns=["as_of_date"])
    r = rank_stocks(feat[feat["date"] <= cp], snap, use_market_regime=False)
    if len(r) < 10:
        return None
    return r.head(10)[["symbol","rank","total_score","market_regime"]].rename(columns={"total_score":"score"})

def run_walk_forward() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        raise RuntimeError("ohlcv.csv is missing")

    hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["date", "symbol", "open", "high", "low", "close"])
    if hist.empty:
        raise RuntimeError("No historical OHLC data available")

    feat = features(hist)
    work = add_targets(feat).dropna(subset=FEATURE_COLUMNS + list(TARGETS.values())).copy()
    dates = pd.Series(pd.to_datetime(feat["date"], errors="coerce").dropna().dt.normalize().unique())
    checkpoints = _checkpoints(dates)
    rows: list[dict] = []
    regime_comparison_rows: list[dict] = []

    for checkpoint in checkpoints:
        train_rows = work[work["date"] < checkpoint]
        if len(train_rows) < MIN_TRAIN_ROWS:
            continue

        session_all = feat[feat["date"] == checkpoint].dropna(subset=FEATURE_COLUMNS).copy()
        if session_all.empty:
            continue

        # Production currently ranks with use_market_regime=False. Keep the
        # primary walk-forward evaluation on that exact ranking path so model
        # evidence is not based on a feature that production does not use.
        production_ranking = _historical_production_selection(feat, checkpoint)
        if production_ranking is None:
            continue
        regime_ranking = rank_stocks(
            feat[feat["date"] <= checkpoint], None, use_market_regime=True
        )
        top = production_ranking.head(10).copy()
        top["production_selected"] = 1
        regime_top = regime_ranking.head(10)[["symbol"]].copy()
        regime_top["regime_selected"] = 1
        session_all = session_all.merge(top, on="symbol", how="left")
        session_all = session_all.merge(regime_top, on="symbol", how="left")
        session_all["production_selected"] = session_all["production_selected"].fillna(0).astype(int)
        session_all["regime_selected"] = session_all["regime_selected"].fillna(0).astype(int)
        session = session_all[session_all["production_selected"] == 1].copy()
        if len(session) < 10:
            continue

        next_dates = dates[dates > checkpoint]
        if next_dates.empty:
            continue
        target_date = next_dates.min()
        actual = hist[hist["date"] == target_date][["symbol", "open", "high", "low", "close"]]
        # Merge actual next-session OHLC explicitly so feature columns keep their names.
        actual = actual.rename(columns={"open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close"})
        session_all = session_all.merge(actual, on="symbol", how="inner")
        session = session_all[session_all["production_selected"] == 1].copy()
        regime_session = session_all[session_all["regime_selected"] == 1].copy()
        if len(session) < 10 or len(regime_session) < 10:
            continue

        # Previous-close baseline is evaluated on the same production-selected
        # basket, so model-vs-baseline comparisons use identical stocks.
        for field in ["open", "high", "low", "close"]:
            session_all[f"baseline_{field}"] = session_all["close"]

        models = _fit_models(train_rows)
        challenger_models = {
            name: _fit_challenger(train_rows[FEATURE_COLUMNS], train_rows[target])
            for name, target in TARGETS.items()
        }
        for name, model in models.items():
            session_all[f"predicted_{name}"] = session_all["close"] * (1 + _predict(model, session_all[FEATURE_COLUMNS]))
            session_all[f"challenger_predicted_{name}"] = session_all["close"] * (
                1 + challenger_models[name].predict(session_all[FEATURE_COLUMNS])
            )
        session = session_all[session_all["production_selected"] == 1].copy()
        regime_session = session_all[session_all["regime_selected"] == 1].copy()
        if len(session) < 10 or len(regime_session) < 10:
            continue

        session["predicted_high"] = session[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
        session["predicted_low"] = session[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)

        # Regime is a separate challenger. Compare its selected basket with the
        # production-selected basket; never use the regime result for the main
        # model promotion evidence.
        production_mape = (
            abs(session["actual_close"] - session["predicted_close"])
            / session["actual_close"].abs()
        ).mean()
        regime_mape = (
            abs(regime_session["actual_close"] - regime_session["predicted_close"])
            / regime_session["actual_close"].abs()
        ).mean()
        regime_relative = _safe_relative_improvement(production_mape, regime_mape)
        regime_comparison_rows.append({
            "prediction_date": checkpoint,
            "target_date": target_date,
            "regime": session["market_regime"].iloc[0],
            "production_selection_close_mape_pct": production_mape * 100,
            "regime_selection_close_mape_pct": regime_mape * 100,
            "relative_improvement_pct": regime_relative,
        })

        for _, row in session.iterrows():
            result = {
                "prediction_date": checkpoint,
                "target_date": target_date,
                "symbol": row["symbol"],
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "base_close": float(row["close"]),
                "market_regime": row.get("market_regime", "NEUTRAL"),
                "regime_selected": int(row.get("regime_selected", 0)),
                "production_selected": int(row.get("production_selected", 0)),
            }
            for field in ["open", "high", "low", "close"]:
                actual_value = float(row[f"actual_{field}"])
                pred = float(row[f"predicted_{field}"])
                challenger = float(row[f"challenger_predicted_{field}"])
                baseline = float(row[f"baseline_{field}"])
                result[f"predicted_{field}"] = pred
                result[f"actual_{field}"] = actual_value
                result[f"{field}_abs_pct_error"] = (
                    abs(actual_value - pred) / abs(actual_value) if actual_value else np.nan
                )
                result[f"challenger_{field}_abs_pct_error"] = (
                    abs(actual_value - challenger) / abs(actual_value) if actual_value else np.nan
                )
                result[f"baseline_{field}_abs_pct_error"] = (
                    abs(actual_value - baseline) / abs(actual_value) if actual_value else np.nan
                )

            predicted_return = result["predicted_close"] / result["base_close"] - 1
            actual_return = result["actual_close"] / result["base_close"] - 1
            result["close_direction_correct"] = int(np.sign(predicted_return) == np.sign(actual_return))
            # Previous-close predicts 0% return, so it has no directional call.
            # Do not score it as wrong on every non-zero market move.
            result["baseline_close_direction_correct"] = np.nan
            rows.append(result)

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No walk-forward checkpoints produced usable results")

    output = output.drop_duplicates(["target_date", "symbol"]).sort_values(["target_date", "rank"])
    output.to_csv(OUTPUT_FILE, index=False)

    summary = output.groupby("target_date").agg(
        stocks=("symbol", "count"),
        open_mape=("open_abs_pct_error", "mean"),
        high_mape=("high_abs_pct_error", "mean"),
        low_mape=("low_abs_pct_error", "mean"),
        close_mape=("close_abs_pct_error", "mean"),
        baseline_open_mape=("baseline_open_abs_pct_error", "mean"),
        baseline_high_mape=("baseline_high_abs_pct_error", "mean"),
        baseline_low_mape=("baseline_low_abs_pct_error", "mean"),
        baseline_close_mape=("baseline_close_abs_pct_error", "mean"),
        close_direction_accuracy=("close_direction_correct", "mean"),
        baseline_close_direction_accuracy=("baseline_close_direction_correct", "mean"),
    ).reset_index()

    for field in ["open", "high", "low", "close"]:
        summary[f"{field}_mape_pct"] = summary[f"{field}_mape"] * 100
        summary[f"baseline_{field}_mape_pct"] = summary[f"baseline_{field}_mape"] * 100

    summary["close_direction_accuracy_pct"] = summary["close_direction_accuracy"] * 100
    # Previous-close has no directional call; report direction accuracy as N/A.
    summary["baseline_close_direction_accuracy_pct"] = np.nan
    summary["close_mape_delta_pct"] = (
        summary["close_mape_pct"] - summary["baseline_close_mape_pct"]
    )
    summary["close_direction_delta_pct_points"] = (
        summary["close_direction_accuracy_pct"]
        - summary["baseline_close_direction_accuracy_pct"]
    )
    summary.to_csv(SUMMARY_FILE, index=False)
    regime_validation = pd.DataFrame(regime_comparison_rows)
    if not regime_validation.empty:
        regime_validation.to_csv(DATA / "regime_validation.csv", index=False)
    regime_summary = _select_model(output)
    regime_summary["regime_validation_sessions"] = len(regime_validation)
    valid_regime = regime_validation[
        np.isfinite(regime_validation["relative_improvement_pct"])
    ] if not regime_validation.empty else pd.DataFrame()
    regime_summary["regime_valid_comparison_sessions"] = len(valid_regime)
    regime_summary["regime_avg_relative_improvement_pct"] = (
        float(valid_regime["relative_improvement_pct"].mean())
        if not valid_regime.empty else np.nan
    )
    regime_summary["regime_promotion_gate_passed"] = bool(
        len(valid_regime) >= MIN_SELECTION_SESSIONS
        and np.isfinite(regime_summary["regime_avg_relative_improvement_pct"].iloc[0])
        and regime_summary["regime_avg_relative_improvement_pct"].iloc[0]
        >= MIN_REGIME_RELATIVE_IMPROVEMENT * 100
    )
    regime_summary.to_csv(SELECTION_FILE, index=False)
    return output


if __name__ == "__main__":
    result = run_walk_forward()
    print(
        f"Walk-forward complete: {len(result)} predictions, "
        f"{result['target_date'].nunique()} sessions, "
        f"model close MAPE={result['close_abs_pct_error'].mean() * 100:.2f}%, "
        f"baseline close MAPE={result['baseline_close_abs_pct_error'].mean() * 100:.2f}%, "
        f"model direction accuracy={result['close_direction_correct'].mean() * 100:.2f}%, "
        f"baseline direction accuracy=N/A (previous-close baseline has no direction)"
    )
