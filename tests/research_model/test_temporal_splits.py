from datetime import date, timedelta

import pytest

from research_model.validation.temporal_splits import purged_walk_forward_splits


def rows_for_days(count=12, horizon=1):
    start = date(2024, 1, 1)
    return [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "target_date": (start + timedelta(days=i + horizon)).isoformat(),
            "symbol": "AAA",
        }
        for i in range(count)
    ]


def test_walk_forward_folds_are_chronological_and_label_purged():
    rows = rows_for_days(12, horizon=1)
    folds = purged_walk_forward_splits(rows, n_splits=3, test_size=2)
    assert len(folds) == 3
    for fold in folds:
        test_start = date.fromisoformat(fold["test_start"])
        for index in fold["train_indices"]:
            assert date.fromisoformat(rows[index]["date"]) < test_start
            assert date.fromisoformat(rows[index]["target_date"]) < test_start
        for index in fold["test_indices"]:
            assert date.fromisoformat(rows[index]["date"]) >= test_start


def test_embargo_removes_training_labels_inside_calendar_gap():
    rows = rows_for_days(12, horizon=1)
    without = purged_walk_forward_splits(rows, n_splits=2, test_size=2)
    with_gap = purged_walk_forward_splits(rows, n_splits=2, test_size=2, embargo_days=2)
    assert len(with_gap[0]["train_indices"]) < len(without[0]["train_indices"])
    cutoff = date.fromisoformat(with_gap[0]["purge_embargo_cutoff"])
    assert all(date.fromisoformat(rows[i]["target_date"]) < cutoff for i in with_gap[0]["train_indices"])


def test_multiple_symbols_on_same_date_stay_in_same_test_fold():
    rows = rows_for_days(10)
    rows.insert(4, {**rows[4], "symbol": "BBB"})
    folds = purged_walk_forward_splits(rows, n_splits=2, test_size=2)
    for fold in folds:
        test_dates = {rows[i]["date"] for i in fold["test_indices"]}
        assert all(rows[i]["date"] in test_dates for i in fold["test_indices"])


@pytest.mark.parametrize(
    "kwargs",
    [{"n_splits": 0}, {"n_splits": 3, "test_size": 0}, {"n_splits": 2, "embargo_days": -1}],
)
def test_invalid_split_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        purged_walk_forward_splits(rows_for_days(), **kwargs)


def test_label_end_before_observation_is_rejected():
    rows = rows_for_days()
    rows[0]["target_date"] = "2023-12-31"
    with pytest.raises(ValueError, match="precedes observation"):
        purged_walk_forward_splits(rows, n_splits=2, test_size=2)


def test_invalid_dates_are_rejected():
    rows = rows_for_days()
    rows[0]["date"] = "not-a-date"
    with pytest.raises(ValueError, match="invalid date"):
        purged_walk_forward_splits(rows, n_splits=2, test_size=2)
