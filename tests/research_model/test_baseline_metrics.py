import pytest

from research_model.validation.baseline_metrics import evaluate_zero_return_baseline


def test_model_and_zero_baseline_use_identical_test_indices():
    rows = [
        {"target_return": 0.10, "prediction": 0.08},
        {"target_return": -0.10, "prediction": -0.05},
        {"target_return": 0.05, "prediction": 0.04},
        {"target_return": -0.02, "prediction": 0.03},
    ]
    folds = [
        {"fold": 1, "train_indices": [0], "test_indices": [1, 2]},
        {"fold": 2, "train_indices": [0, 1], "test_indices": [3]},
    ]
    report = evaluate_zero_return_baseline(rows, folds)
    assert report["evaluation_scope"] == "OUT_OF_SAMPLE_TEST_FOLDS_ONLY"
    assert report["aggregate"]["model"]["observations"] == 3
    assert report["aggregate"]["zero_return_baseline"]["observations"] == 3
    assert report["aggregate"]["model"]["mae"] < report["aggregate"]["zero_return_baseline"]["mae"]
    assert report["training_authorized"] is False
    assert report["trading_simulation_implemented"] is False


def test_test_indices_cannot_overlap_between_folds():
    rows = [{"target_return": 0.1, "prediction": 0.1}]
    folds = [{"test_indices": [0]}, {"test_indices": [0]}]
    with pytest.raises(ValueError, match="overlapping/duplicate"):
        evaluate_zero_return_baseline(rows, folds)


def test_duplicate_test_index_within_fold_rejected():
    rows = [{"target_return": 0.1, "prediction": 0.1}]
    with pytest.raises(ValueError, match="overlapping/duplicate"):
        evaluate_zero_return_baseline(rows, [{"test_indices": [0, 0]}])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "not-a-number"])
def test_non_finite_or_non_numeric_inputs_rejected(bad):
    rows = [{"target_return": 0.1, "prediction": bad}]
    with pytest.raises(ValueError, match="finite numeric"):
        evaluate_zero_return_baseline(rows, [{"test_indices": [0]}])


def test_empty_fold_and_out_of_range_index_rejected():
    rows = [{"target_return": 0.1, "prediction": 0.1}]
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_zero_return_baseline(rows, [{"test_indices": []}])
    with pytest.raises(ValueError, match="out of range"):
        evaluate_zero_return_baseline(rows, [{"test_indices": [1]}])

def test_training_rows_do_not_need_predictions_or_enter_metrics():
    rows = [
        {"target_return": 0.9},
        {"target_return": 0.1, "prediction": 0.08},
    ]
    report = evaluate_zero_return_baseline(
        rows,
        [{"train_indices": [0], "test_indices": [1]}],
    )
    assert report["aggregate"]["model"]["observations"] == 1
    assert report["aggregate"]["model"]["mean_realized_return"] == pytest.approx(0.1)
