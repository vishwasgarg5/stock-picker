import pytest

from research_model.validation.target_contract import (
    TargetContract,
    validate_target_contract,
)


def base_contract(**overrides):
    values = {
        "return_basis": "split_adjusted_price",
        "horizon_sessions": 5,
        "decision_price_field": "close",
        "entry_price_field": "next_open",
        "exit_price_field": "horizon_close",
        "label_end_field": "target_date",
        "dividend_treatment": "excluded",
        "corporate_action_evidence": "UNVERIFIED",
    }
    values.update(overrides)
    return TargetContract(**values)


def test_price_only_contract_must_explicitly_exclude_dividends():
    contract = base_contract()
    result = validate_target_contract(contract)
    assert result["return_basis"] == "split_adjusted_price"
    assert result["dividend_treatment"] == "excluded"
    assert result["training_authorized"] == 0
    assert result["label_generation_implemented"] == 0


def test_dividend_inclusive_contract_requires_verified_actions_and_policy():
    with pytest.raises(ValueError, match="independently verified"):
        base_contract(
            return_basis="cash_dividend_inclusive",
            dividend_treatment="cash_received_no_reinvestment",
        )
    accepted = base_contract(
        return_basis="cash_dividend_inclusive",
        dividend_treatment="cash_received_no_reinvestment",
        corporate_action_evidence="INDEPENDENTLY_VERIFIED",
    )
    assert validate_target_contract(accepted)["return_basis"] == "cash_dividend_inclusive"


def test_return_basis_cannot_silently_mix_dividend_treatment():
    with pytest.raises(ValueError, match="requires dividend_treatment"):
        base_contract(dividend_treatment="cash_received_no_reinvestment")


@pytest.mark.parametrize("horizon", [0, -1, True])
def test_invalid_horizon_fails_closed(horizon):
    with pytest.raises(ValueError, match="positive integer"):
        base_contract(horizon_sessions=horizon)


def test_unknown_return_basis_fails_closed():
    with pytest.raises(ValueError, match="unsupported return_basis"):
        base_contract(return_basis="adjusted_total_return")


def test_non_integer_horizon_fails_closed():
    with pytest.raises(ValueError, match="positive integer"):
        base_contract(horizon_sessions="5")


def test_missing_price_field_fails_closed():
    with pytest.raises(ValueError, match="entry_price_field"):
        base_contract(entry_price_field=None)
