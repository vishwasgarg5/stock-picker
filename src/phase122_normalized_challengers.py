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

from .pipeline import FEATURE_COLUMNS, _ensemble_predict, _fit_ensemble

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


def build_point_in_time_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build causal features without retrospectively applying future stock splits.

    At each split date, only the already-observed rolling price/volume state is
    restated into the new share units. A split occurring after date t cannot
    alter any feature row at or before t.
    """
    missing = sorted(REQUIRED_HISTORY.difference(history.columns))
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")
    x = history.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    if "stock_splits" not in x:
        x["stock_splits"] = 0.0
    x["stock_splits"] = pd.to_numeric(x["stock_splits"], errors="coerce").fillna(0.0)
    if not np.isfinite(x["stock_splits"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite stock split ratio in history")
    x = x.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    rows: list[dict[str, Any]] = []

    for symbol, group in x.groupby("symbol", sort=False):
        price_buffer: list[float] = []
        volume_buffer: list[float] = []
        true_range_buffer: list[float] = []
        return_buffer: list[float] = []
        ema12 = ema20 = ema26 = ema50 = macd_signal = None
        alpha12, alpha20 = 2.0 / 13.0, 2.0 / 21.0
        alpha26, alpha50, alpha9 = 2.0 / 27.0, 2.0 / 51.0, 2.0 / 10.0

        for row in group.itertuples(index=False):
            close = float(row.close)
            high = float(row.high)
            low = float(row.low)
            volume = float(row.volume)
            split_ratio = float(row.stock_splits) if np.isfinite(row.stock_splits) else 0.0
            if not all(np.isfinite(v) and v > 0 for v in (close, high, low, volume)):
                continue
            if split_ratio < 0:
                raise ValueError(f"Invalid split ratio for {symbol} on {row.date}")
            if split_ratio > 0:
                price_buffer = [value / split_ratio for value in price_buffer]
                true_range_buffer = [value / split_ratio for value in true_range_buffer]
                volume_buffer = [value * split_ratio for value in volume_buffer]
                ema12 = ema12 / split_ratio if ema12 is not None else None
                ema20 = ema20 / split_ratio if ema20 is not None else None
                ema26 = ema26 / split_ratio if ema26 is not None else None
                ema50 = ema50 / split_ratio if ema50 is not None else None
                macd_signal = macd_signal / split_ratio if macd_signal is not None else None

            previous_close = price_buffer[-1] if price_buffer else np.nan
            daily_return = close / previous_close - 1.0 if np.isfinite(previous_close) and previous_close > 0 else np.nan
            true_range = max(
                high - low,
                abs(high - previous_close) if np.isfinite(previous_close) else 0.0,
                abs(low - previous_close) if np.isfinite(previous_close) else 0.0,
            )
            price_buffer.append(close)
            volume_buffer.append(volume)
            true_range_buffer.append(true_range)
            return_buffer.append(daily_return)
            # Retain enough history for the 50-session moving average and EMA warmup.
            price_buffer = price_buffer[-250:]
            volume_buffer = volume_buffer[-20:]
            true_range_buffer = true_range_buffer[-14:]
            return_buffer = return_buffer[-20:]

            ema12 = close if ema12 is None else alpha12 * close + (1.0 - alpha12) * ema12
            ema20 = close if ema20 is None else alpha20 * close + (1.0 - alpha20) * ema20
            ema26 = close if ema26 is None else alpha26 * close + (1.0 - alpha26) * ema26
            ema50 = close if ema50 is None else alpha50 * close + (1.0 - alpha50) * ema50
            macd = ema12 - ema26
            macd_signal = macd if macd_signal is None else alpha9 * macd + (1.0 - alpha9) * macd_signal

            sma20 = float(np.mean(price_buffer[-20:])) if len(price_buffer) >= 20 else np.nan
            sma50 = float(np.mean(price_buffer[-50:])) if len(price_buffer) >= 50 else np.nan
            std20 = float(np.std(price_buffer[-20:], ddof=1)) if len(price_buffer) >= 20 else np.nan
            recent_returns = np.asarray(return_buffer[-20:], dtype=float)
            valid_returns = recent_returns[np.isfinite(recent_returns)]
            volatility20 = float(np.std(valid_returns, ddof=1)) if len(valid_returns) >= 20 else np.nan
            deltas = np.diff(np.asarray(price_buffer[-15:], dtype=float))
            if len(deltas) >= 14:
                gain = float(np.mean(np.maximum(deltas[-14:], 0.0)))
                loss = float(np.mean(np.maximum(-deltas[-14:], 0.0)))
                rsi14 = float(100.0 - 100.0 / (1.0 + gain / loss)) if loss > 0 else np.nan
            else:
                rsi14 = np.nan
            volume_ma20 = float(np.mean(volume_buffer[-20:])) if len(volume_buffer) >= 20 else np.nan
            volume_ma5 = float(np.mean(volume_buffer[-5:])) if len(volume_buffer) >= 5 else np.nan

            rows.append({
                "date": pd.Timestamp(row.date), "symbol": symbol,
                "open": float(row.open), "high": high, "low": low, "close": close,
                "volume": volume, "split_adjusted_close": close,
                "return_1d": daily_return,
                "return_5d": close / price_buffer[-6] - 1.0 if len(price_buffer) >= 6 else np.nan,
                "return_20d": close / price_buffer[-21] - 1.0 if len(price_buffer) >= 21 else np.nan,
                "sma20": sma20, "sma50": sma50, "ema20": ema20, "ema50": ema50,
                "rsi14": rsi14,
                "volume_ratio": volume / volume_ma20 if np.isfinite(volume_ma20) and volume_ma20 > 0 else np.nan,
                "atr14_pct": float(np.mean(true_range_buffer[-14:]) / close) if len(true_range_buffer) >= 14 else np.nan,
                "macd": macd, "macd_signal": macd_signal,
                "bb_position": (close - sma20) / (2.0 * std20) if np.isfinite(std20) and std20 > 0 else np.nan,
                "range_pct": (high - low) / close,
                "close_sma20_gap": close / sma20 - 1.0 if np.isfinite(sma20) and sma20 > 0 else np.nan,
                "close_sma50_gap": close / sma50 - 1.0 if np.isfinite(sma50) and sma50 > 0 else np.nan,
                "volatility20": volatility20,
                "volume_trend5": volume_ma5 / volume_ma20 if np.isfinite(volume_ma5) and np.isfinite(volume_ma20) and volume_ma20 > 0 else np.nan,
                "split_ratio": split_ratio,
            })

    result = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    # The next close is converted into the current share units using only the
    # split ratio observed on that next session. This is the realized label,
    # never a feature available to the prediction.
    grouped = result.groupby("symbol", sort=False)
    next_close = grouped["close"].shift(-1)
    next_split = grouped["split_ratio"].shift(-1).fillna(0.0)
    next_date = grouped["date"].shift(-1)
    split_multiplier = next_split.where(next_split.gt(0), 1.0)
    result["target_date"] = next_date
    result["target_close_return"] = (next_close * split_multiplier) / result["close"] - 1.0
    return result


def build_normalized_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build causal features, replacing price-level indicators with ratios."""
    x = build_point_in_time_features(history)
    close = pd.to_numeric(x["close"], errors="coerce").replace(0, np.nan)
    for source, destination in NORMALIZED_REPLACEMENTS.items():
        x[destination] = pd.to_numeric(x[source], errors="coerce") / close
        if source.startswith(("sma", "ema")):
            x[destination] = x[destination] - 1.0
    return x


def _metrics(actual_return: pd.Series, predicted_return: pd.Series) -> dict[str, float | int | None]:
    actual = pd.to_numeric(actual_return, errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(predicted_return, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > -1.0)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return {
            "rows": 0, "invalid_nonpositive_price_forecasts": 0, "mape_pct": None, "directional_accuracy_pct": None,
            "signed_bias_pct_points": None, "mean_abs_predicted_move_pct": None,
            "mean_abs_realized_move_pct": None, "forecast_to_realized_magnitude_ratio": None,
        }
    actual_close = 1.0 + actual
    # A return forecast below -100% implies a non-positive price. Score it as
    # a zero-price forecast rather than silently dropping that test observation.
    predicted_close = np.maximum(1.0 + predicted, 0.0)
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
            if not np.isfinite(np.asarray(predicted, dtype=float)).all():
                raise RuntimeError(f"Non-finite predictions from {name} in fold {fold_number}")
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
        "status": ("SHADOW_EXPERIMENT_COMPLETE" if len(fold_rows) == n_splits else ("INCOMPLETE_CHRONOLOGICAL_FOLDS" if fold_rows else "INSUFFICIENT_CHRONOLOGICAL_DATA")),
        "production_model_changed": False,
        "production_champion": "V1",
        "challengers_shadow_only": True,
        "automatic_promotion": False,
        "feature_transforms": {
            "sma20_sma50_ema20_ema50": "indicator / contemporaneous split-adjusted close - 1",
            "macd_macd_signal": "indicator / contemporaneous split-adjusted close",
            "other_features": "causally recomputed from observations available through the feature date",
        },
        "regularized_model": {"estimator": "StandardScaler + Ridge", "alpha": float(ridge_alpha)},
        "baseline": "zero next-session return; predicted close equals contemporaneous split-adjusted close",
        "fold_policy": {
            "type": "expanding_window",
            "initial_train_fraction": float(initial_train_fraction),
            "requested_folds": int(n_splits),
            "min_train_rows": int(min_train_rows),
            "training_label_rule": "target_date strictly earlier than first prediction_date in each test fold",
            "feature_point_in_time_rule": "Rolling price/volume state is adjusted only when a split is observed on or before that feature date; future split events cannot rewrite prior features.",
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
            "All feature rows use point-in-time split handling; later corporate actions do not restate earlier feature values.",
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
    output_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_from_csv()
    print(json.dumps(result, indent=2, default=str))
