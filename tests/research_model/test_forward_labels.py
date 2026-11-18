import pytest

from research_model.validation.forward_labels import generate_forward_labels
from research_model.validation.target_contract import TargetContract


def contract(**overrides):
    values = {
        "return_basis": "split_adjusted_price",
        "horizon_sessions": 2,
        "decision_price_field": "close",
        "entry_price_field": "next_open",
        "exit_price_field": "close",
        "label_end_field": "target_date",
        "dividend_treatment": "excluded",
        "corporate_action_evidence": "INDEPENDENTLY_VERIFIED",
    }
    values.update(overrides)
    return TargetContract(**values)


def test_labels_use_n_sessions_ahead_within_symbol_and_emit_target_end():
    rows = [
        {"symbol": "AAA", "date": "2024-01-03", "close": 10, "next_open": 11},
        {"symbol": "BBB", "date": "2024-01-01", "close": 20, "next_open": 21},
        {"symbol": "AAA", "date": "2024-01-01", "close": 8, "next_open": 9},
        {"symbol": "AAA", "date": "2024-01-02", "close": 9, "next_open": 10},
        {"symbol": "AAA", "date": "2024-01-04", "close": 12, "next_open": 13},
    ]
    labels = generate_forward_labels(rows, contract())
    assert len(labels) == 2
    first = labels[0]
    assert first["symbol"] == "AAA"
    assert first["decision_date"] == "2024-01-01"
    assert first["target_date"] == "2024-01-03"
    assert first["target_return"] == pytest.approx(10 / 9 - 1)
    assert first["label_status"].startswith("SYNTHETIC_OR_CALLER_SUPPLIED")


def test_trailing_incomplete_horizon_is_omitted():
    rows = [
        {"symbol": "AAA", "date": f"2024-01-0{i}", "close": 10+i, "next_open": 10+i}
        for i in range(1, 4)
    ]
    assert len(generate_forward_labels(rows, contract())) == 1


def test_duplicate_symbol_date_fails_closed():
    row = {"symbol": "AAA", "date": "2024-01-01", "close": 10, "next_open": 10}
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        generate_forward_labels([row, row], contract())


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf"), True])
def test_invalid_price_fails_closed(price):
    rows = [
        {"symbol": "AAA", "date": "2024-01-01", "close": 10, "next_open": 10},
        {"symbol": "AAA", "date": "2024-01-02", "close": price, "next_open": 10},
        {"symbol": "AAA", "date": "2024-01-03", "close": 12, "next_open": 10},
    ]
    with pytest.raises(ValueError, match="finite positive"):
        generate_forward_labels(rows, contract(horizon_sessions=1))


def test_cash_dividend_inclusive_generation_is_not_approximated():
    cash_contract = contract(
        return_basis="cash_dividend_inclusive",
        dividend_treatment="cash_received_no_reinvestment",
    )
    with pytest.raises(ValueError, match="not implemented"):
        generate_forward_labels([], cash_contract)


def test_labels_are_deterministic_under_input_row_reordering():
    rows = [
        {"symbol": "AAA", "date": f"2024-01-0{i}", "close": 10+i, "next_open": 10+i}
        for i in range(1, 5)
    ]
    assert generate_forward_labels(rows, contract()) == generate_forward_labels(
        list(reversed(rows)), contract()
    )
