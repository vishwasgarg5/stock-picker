from __future__ import annotations

"""Phase 122: shadow-only chronological forecast challenger experiment.

This module never writes production model artifacts or changes ranking. It
compares the incumbent ensemble, a scale-normalized ensemble, a regularized
return model, and a zero-return/previous-close baseline on identical folds.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import FEATURE_COLUMNS, _ensemble_predict, _fit_ensemble, add_targets, features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_INPUT = DATA / "ohlcv.csv"
DEFAULT_REPORT = DATA / "phase122_normalized_challenger_report.json"

NORMALIZED_REPLACEMENTS = {
    "sma20": "sma20_rel_close",
    "sma50": "sma50_rel_close",
    "ema20": "ema20_rel_close",
    "ema50": "ema50_rel_close",
    "macd": "macd_rel_close",
    "macd_signal": "macd_signal_rel_close",
}
NORMALIZED_FEATURE_COLUMNS = [
    NORMALIZED_REPLACEMENTS.get(column, column) for column in FEATURE_COLUMNS
]
REQUIRED_HISTORY = {"date", "symbol", "open", "high", "low", "close", "volume"}


def build_normalized_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build existing causal features, replacing price-level indicators by ratios.

    The denominator is each row's contemporaneous split-adjusted close. No
    cross-row/global statistic is used in this transformation.
    """
    missing = sorted(REQUIRED_HISTORY.difference(history.columns))
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")
    x = features(history.copy())
    close = pd.to_numeric(x["split_adjusted_close"], errors="coerce").replace(0, np.nan)
    for source, destination in NORMALIZED_REPLACEMENTS.items():
        x[destination] = pd.to_numeric(x[source], errors="coerce") / close
        if source.startswith(("sma", "ema")):
            x[destination] = x[destination] - 1.0
    return x


def _metrics(actual_return: pd.Series, predicted_return: pd.Series) -> dict[str, float | int | None]:
    actual = pd.to_numeric(actual_return, errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(predicted_return, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > -1.0) & (predicted > -1.0)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return {
            "rows": 0, "mape_pct": None, "directional_accuracy_pct": None,
            "signed_bias_pct_points": None, "mean_abs_predicted_move_pct": None,
            "mean_abs_realized_move_pct": None, "forecast_to_realized_magnitude_ratio": None,
        }
    actual_close = 1.0 + actual
    predicted_close = 1.0 + predicted
    mape = float(np.mean(np.abs(predicted_close - actual_close) / actual_close) * 100.0)
    direction = float(np.mean(np.sign(predicted) == np.sign(actual)) * 100.0)
    mean_abs_pred = float(np.mean(np.abs(predicted)) * 100.0)
    mean_abs_actual = float(np.mean(np.abs(actual)) * 100.0)
    return {
        "rows": int(len(actual)),
        "mape_pct": mape,
        "directional_accuracy_pct": direction,
        "signed_bias_pct_points": float(np.mean(predicted - actual) * 100.0),
        "mean_abs_predicted_move_pct": mean_abs_pred,
        "mean_abs_realized_move_pct": mean_abs_actual,
        "forecast_to_realized_magnitude_ratio": (
            float(mean_abs_pred / mean_abs_actual) if mean_abs_actual > 0 else None
        ),
    }


def _chronological_folds(
    dates: list[pd.Timestamp], initial_train_fraction: float, n_splits: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]]:
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must be between 0 and 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    first_test = max(1, int(len(dates) * initial_train_fraction))
    remaining = dates[first_test:]
    if len(remaining) < n_splits:
        raise ValueError("Not enough distinct dates for requested chronological folds")
    chunks = [chunk for chunk in np.array_split(np.asarray(remaining, dtype=object), n_splits) if len(chunk)]
    folds = []
    for chunk in chunks:
        test_dates = [pd.Timestamp(value) for value in chunk.tolist()]
        folds.append((pd.Timestamp(test_dates[0]), pd.Timestamp(test_dates[-1]), test_dates))
    return folds


def run_challenger_experiment(
    history: pd.DataFrame,
    *,
    initial_train_fraction: float = 0.60,
    n_splits: int = 4,
    min_train_rows: int = 500,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    """Run expanding-window evaluation with a strict label-availability cutoff."""
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    raw = history.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["symbol"] = raw["symbol"].astype(str).str.strip().str.upper()
    raw = raw.dropna(subset=["date", "symbol"])
    if raw.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/symbol history keys; refusing ambiguous replay")
    x = build_normalized_features(raw)
    x = add_targets(x)
    x["target_date"] = x.groupby("symbol", sort=False)["date"].shift(-1)
    x["target_close_return"] = pd.to_numeric(x["target_close_return"], errors="coerce")
    numeric = list(FEATURE_COLUMNS) + NORMALIZED_FEATURE_COLUMNS + ["target_close_return"]
    for column in numeric:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["date", "target_date", "split_adjusted_close", "target_close_return"])
    x = x[x["target_date"] > x["date"]].copy()
    dates = sorted(pd.Timestamp(value) for value in x["date"].unique())
    folds = _chronological_folds(dates, initial_train_fraction, n_splits)

    fold_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[pd.DataFrame]] = {
        "incumbent_control": [], "normalized_ensemble": [],
        "regularized_return": [], "previous_close_baseline": [],
    }
    skipped_folds: list[dict[str, Any]] = []
    for fold_number, (test_start, test_end, test_dates) in enumerate(folds, start=1):
        # Critical no-lookahead rule: the label's realized target session must
        # be strictly earlier than the first prediction date in this fold.
        train = x.loc[x["target_date"] < test_start].copy()
        test = x.loc[x["date"].isin(test_dates)].copy()
        train = train.dropna(subset=FEATURE_COLUMNS + NORMALIZED_FEATURE_COLUMNS + ["target_close_return"])
        test = test.dropna(subset=FEATURE_COLUMNS + NORMALIZED_FEATURE_COLUMNS + ["target_close_return"])
        if len(train) < min_train_rows or test.empty:
            skipped_folds.append({
                "fold": fold_number, "test_start": str(test_start.date()),
                "test_end": str(test_end.date()), "training_rows": int(len(train)),
                "test_rows": int(len(test)), "reason": "insufficient_training_or_test_rows",
            })
            continue
        if not (pd.to_datetime(train["target_date"]) < test_start).all():
            raise RuntimeError("No-lookahead audit failed: training label reaches test prediction date")
        y_train = train["target_close_return"]
        y_test = test["target_close_return"]
        base = pd.to_numeric(test["split_adjusted_close"], errors="coerce")
        fold_models: dict[str, np.ndarray] = {
            "previous_close_baseline": np.zeros(len(test), dtype=float)
        }

        incumbent = _fit_ensemble(train[FEATURE_COLUMNS], y_train)
        incumbent_pred, _ = _ensemble_predict(incumbent, test[FEATURE_COLUMNS])
        fold_models["incumbent_control"] = incumbent_pred

        normalized = _fit_ensemble(train[NORMALIZED_FEATURE_COLUMNS], y_train)
        normalized_pred, _ = _ensemble_predict(normalized, test[NORMALIZED_FEATURE_COLUMNS])
        fold_models["normalized_ensemble"] = normalized_pred

        regularized = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        regularized.fit(train[NORMALIZED_FEATURE_COLUMNS], y_train)
        ridge_pred = regularized.predict(test[NORMALIZED_FEATURE_COLUMNS])
        fold_models["regularized_return"] = ridge_pred

        fold_detail: dict[str, Any] = {
            "fold": fold_number,
            "train_prediction_dates": int(train["date"].nunique()),
            "train_rows": int(len(train)),
            "latest_training_target_date": str(pd.Timestamp(train["target_date"].max()).date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "test_prediction_dates": int(test["date"].nunique()),
            "test_target_dates": int(test["target_date"].nunique()),
            "test_rows": int(len(test)),
            "no_lookahead": bool((pd.to_datetime(train["target_date"]) < test_start).all()),
            "models": {},
        }
        for name, predicted in fold_models.items():
            fold_detail["models"][name] = _metrics(y_test, pd.Series(predicted, index=y_test.index))
            part = test[["date", "target_date", "symbol", "split_adjusted_close"]].copy()
            part["actual_return"] = y_test.to_numpy()
            part["predicted_return"] = np.asarray(predicted, dtype=float)
            part["model"] = name
            predictions[name].append(part)
        fold_rows.append(fold_detail)

    aggregate: dict[str, Any] = {}
    for name, parts in predictions.items():
        if parts:
            joined = pd.concat(parts, ignore_index=True)
            aggregate[name] = _metrics(joined["actual_return"], joined["predicted_return"])
        else:
            aggregate[name] = _metrics(pd.Series(dtype=float), pd.Series(dtype=float))

    fold_metrics: dict[str, dict[str, Any]] = {}
    for name in predictions:
        values = [
            fold["models"][name]["mape_pct"]
            for fold in fold_rows
            if fold["models"][name]["mape_pct"] is not None
        ]
        fold_metrics[name] = {
            "folds_with_results": len(values),
            "fold_mape_mean_pct": float(np.mean(values)) if values else None,
            "fold_mape_std_pct": float(np.std(values)) if values else None,
            "folds_beating_previous_close_mape": sum(
                fold["models"][name]["mape_pct"] < fold["models"]["previous_close_baseline"]["mape_pct"]
                for fold in fold_rows
                if fold["models"][name]["mape_pct"] is not None
                and fold["models"]["previous_close_baseline"]["mape_pct"] is not None
            ),
        }

    return {
        "phase": 122,
        "title": "Scale-normalized and regularized return challengers",
        "status": "SHADOW_EXPERIMENT_COMPLETE" if fold_rows else "INSUFFICIENT_CHRONOLOGICAL_DATA",
        "production_model_changed": False,
        "production_champion": "V1",
        "challengers_shadow_only": True,
        "automatic_promotion": False,
        "feature_transforms": {
            "sma20_sma50_ema20_ema50": "indicator / contemporaneous split-adjusted close - 1",
            "macd_macd_signal": "indicator / contemporaneous split-adjusted close",
            "other_features": "retained from incumbent feature builder",
        },
        "regularized_model": {"estimator": "StandardScaler + Ridge", "alpha": float(ridge_alpha)},
        "baseline": "zero next-session return; predicted close equals contemporaneous split-adjusted close",
        "fold_policy": {
            "type": "expanding_window",
            "initial_train_fraction": float(initial_train_fraction),
            "requested_folds": int(n_splits),
            "min_train_rows": int(min_train_rows),
            "training_label_rule": "target_date strictly earlier than first prediction_date in each test fold",
            "test_dates_are_disjoint": True,
        },
        "folds": fold_rows,
        "skipped_folds": skipped_folds,
        "aggregate_metrics": aggregate,
        "fold_stability": fold_metrics,
        "diagnostics": {
            "incumbent_forecast_to_realized_magnitude_ratio": aggregate["incumbent_control"]["forecast_to_realized_magnitude_ratio"],
            "normalized_forecast_to_realized_magnitude_ratio": aggregate["normalized_ensemble"]["forecast_to_realized_magnitude_ratio"],
            "regularized_forecast_to_realized_magnitude_ratio": aggregate["regularized_return"]["forecast_to_realized_magnitude_ratio"],
            "interpretation": "Magnitude ratio near zero indicates forecast compression. Ratios above one indicate forecast dispersion exceeds realized movement; neither is automatically desirable.",
        },
        "limitations": [
            "This historical replay is exploratory and does not replace fresh prospective validation.",
            "No hyperparameter search is performed on test folds; Ridge alpha is fixed before evaluation.",
            "No volatility regime or symbol exclusions are used to select results.",
            "The incumbent ensemble is re-fit only on each fold's permitted training data as the chronological control.",
        ],
    }


def run_from_csv(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_REPORT,
    **kwargs: Any,
) -> dict[str, Any]:
    history = pd.read_csv(input_path, low_memory=False)
    report = run_challenger_experiment(history, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str) + "\\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_from_csv()
    print(json.dumps(result, indent=2, default=str))
