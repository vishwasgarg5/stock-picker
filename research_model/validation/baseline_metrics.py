"""Out-of-sample baseline metrics on identical chronological test folds.

This module evaluates caller-supplied predictions only. It does not fit models,
select thresholds, simulate trades, or authorize training/production.
"""
from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Mapping, Sequence


def _finite_number(value: Any, *, field: str, row_index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"row {row_index}: {field} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index}: {field} must be finite numeric") from exc
    if not isfinite(number):
        raise ValueError(f"row {row_index}: {field} must be finite numeric")
    return number


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int | bool]:
    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = sqrt(sum(error * error for error in errors) / len(errors))
    directional = sum(
        (prediction > 0 and target > 0) or (prediction < 0 and target < 0)
        or (prediction == 0 and target == 0)
        for target, prediction in zip(actual, predicted)
    ) / len(actual)
    return {
        "observations": len(actual),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional,
        "mean_prediction": sum(predicted) / len(predicted),
        "mean_realized_return": sum(actual) / len(actual),
    }


def evaluate_zero_return_baseline(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    *,
    actual_column: str = "target_return",
    prediction_column: str = "prediction",
) -> dict[str, Any]:
    """Compare model forecasts with a zero-return forecast on each test fold.

    Each fold's test_indices are the sole observations evaluated for that fold.
    The zero-return baseline is scored on the exact same rows. Fold test sets
    must be disjoint to prevent accidental double-counting in aggregate metrics.
    """
    if not rows:
        raise ValueError("rows cannot be empty")
    if not folds:
        raise ValueError("folds cannot be empty")

    parsed: dict[int, tuple[float, float]] = {}
    for index, row in enumerate(rows):
        if actual_column not in row or prediction_column not in row:
            raise ValueError(
                f"row {index}: required {actual_column!r} and {prediction_column!r}"
            )
        parsed[index] = (
            _finite_number(row[actual_column], field=actual_column, row_index=index),
            _finite_number(row[prediction_column], field=prediction_column, row_index=index),
        )

    seen_test_indices: set[int] = set()
    fold_results: list[dict[str, Any]] = []
    all_actual: list[float] = []
    all_predictions: list[float] = []
    for fold_position, fold in enumerate(folds, start=1):
        indices = fold.get("test_indices")
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError(f"fold {fold_position}: test_indices must be non-empty")
        actual: list[float] = []
        predictions: list[float] = []
        local_seen: set[int] = set()
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"fold {fold_position}: test index must be an integer")
            if index < 0 or index >= len(rows):
                raise ValueError(f"fold {fold_position}: test index {index} out of range")
            if index in local_seen or index in seen_test_indices:
                raise ValueError(f"fold {fold_position}: overlapping/duplicate test index {index}")
            local_seen.add(index)
            seen_test_indices.add(index)
            target, prediction = parsed[index]
            actual.append(target)
            predictions.append(prediction)
        model_metrics = _metrics(actual, predictions)
        zero_metrics = _metrics(actual, [0.0] * len(actual))
        fold_results.append({
            "fold": fold.get("fold", fold_position),
            "test_indices": list(indices),
            "model": model_metrics,
            "zero_return_baseline": zero_metrics,
            "model_beats_zero_baseline_mae": model_metrics["mae"] < zero_metrics["mae"],
        })
        all_actual.extend(actual)
        all_predictions.extend(predictions)

    model_aggregate = _metrics(all_actual, all_predictions)
    zero_aggregate = _metrics(all_actual, [0.0] * len(all_actual))
    return {
        "evaluation_scope": "OUT_OF_SAMPLE_TEST_FOLDS_ONLY",
        "baseline": "ZERO_RETURN_FORECAST",
        "folds": fold_results,
        "aggregate": {
            "model": model_aggregate,
            "zero_return_baseline": zero_aggregate,
            "model_beats_zero_baseline_mae": (
                model_aggregate["mae"] < zero_aggregate["mae"]
            ),
        },
        "training_authorized": False,
        "trading_simulation_implemented": False,
    }
