"""Leakage-safe expanding chronological splits for research labels.

Rows are supplied as mappings. Returned indices preserve the caller's original
row order and avoid coupling this capability to a dataframe library.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence


def _as_date(value: Any, *, field: str, row_index: int) -> date:
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value)
        if len(text) < 10:
            raise ValueError("date must contain YYYY-MM-DD")
        return date.fromisoformat(text[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index}: invalid {field} date {value!r}") from exc


def purged_walk_forward_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 3,
    test_size: int | None = None,
    embargo_days: int = 0,
    date_column: str = "date",
    label_end_column: str = "target_date",
) -> list[dict[str, Any]]:
    """Build expanding train / forward test folds with label-overlap purging.

    date_column is the observation/decision date. label_end_column is the last
    date consumed to form that row's target. Training labels must end strictly
    before the first test date minus the calendar-day embargo.

    Test folds are contiguous blocks of unique observation dates. Training
    folds contain only earlier observations; test rows are never shuffled.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if embargo_days < 0:
        raise ValueError("embargo_days cannot be negative")
    if not rows:
        raise ValueError("rows cannot be empty")

    observation_dates: list[date] = []
    label_end_dates: list[date] = []
    for index, row in enumerate(rows):
        if date_column not in row or label_end_column not in row:
            raise ValueError(
                f"row {index}: required columns {date_column!r} and {label_end_column!r}"
            )
        observed = _as_date(row[date_column], field=date_column, row_index=index)
        label_end = _as_date(row[label_end_column], field=label_end_column, row_index=index)
        if label_end < observed:
            raise ValueError(
                f"row {index}: label end {label_end} precedes observation date {observed}"
            )
        observation_dates.append(observed)
        label_end_dates.append(label_end)

    unique_dates = sorted(set(observation_dates))
    if test_size is None:
        test_size = len(unique_dates) // (n_splits + 1)
    if test_size < 1:
        raise ValueError("not enough unique dates for the requested split count")
    first_test_offset = len(unique_dates) - n_splits * test_size
    if first_test_offset < 1:
        raise ValueError("splits must leave at least one initial training date")

    folds: list[dict[str, Any]] = []
    for fold_number in range(n_splits):
        start = first_test_offset + fold_number * test_size
        test_dates = unique_dates[start : start + test_size]
        if len(test_dates) != test_size:
            raise ValueError("not enough dates to build all requested test folds")
        test_start = test_dates[0]
        cutoff = test_start - timedelta(days=embargo_days)
        train_indices = [
            index for index, observed in enumerate(observation_dates)
            if observed < test_start and label_end_dates[index] < cutoff
        ]
        test_indices = [
            index for index, observed in enumerate(observation_dates)
            if test_dates[0] <= observed <= test_dates[-1]
        ]
        if not train_indices or not test_indices:
            raise ValueError(f"fold {fold_number + 1} has an empty train or test set")
        folds.append({
            "fold": fold_number + 1,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_start": min(observation_dates[i] for i in train_indices).isoformat(),
            "train_end": max(observation_dates[i] for i in train_indices).isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_dates[-1].isoformat(),
            "purge_embargo_cutoff": cutoff.isoformat(),
            "embargo_days": embargo_days,
        })
    return folds
