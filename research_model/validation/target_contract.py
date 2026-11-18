"""Explicit, fail-closed research target semantics.

This module defines and validates target configuration only. It deliberately
does not generate returns or labels until callers provide independently
verified corporate-action inputs and an explicit execution convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ReturnBasis = Literal["split_adjusted_price", "cash_dividend_inclusive"]
DividendTreatment = Literal["excluded", "cash_received_no_reinvestment"]
EvidenceStatus = Literal["UNVERIFIED", "INDEPENDENTLY_VERIFIED"]


@dataclass(frozen=True)
class TargetContract:
    """Required semantics for a forward-return target; no implicit defaults."""

    return_basis: ReturnBasis
    horizon_sessions: int
    decision_price_field: str
    entry_price_field: str
    exit_price_field: str
    label_end_field: str
    dividend_treatment: DividendTreatment
    corporate_action_evidence: EvidenceStatus

    def __post_init__(self) -> None:
        if self.return_basis not in {"split_adjusted_price", "cash_dividend_inclusive"}:
            raise ValueError(f"unsupported return_basis: {self.return_basis!r}")
        if isinstance(self.horizon_sessions, bool) or self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be a positive integer")
        for name in (
            "decision_price_field",
            "entry_price_field",
            "exit_price_field",
            "label_end_field",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.corporate_action_evidence not in {
            "UNVERIFIED", "INDEPENDENTLY_VERIFIED"
        }:
            raise ValueError("unsupported corporate_action_evidence status")

        if self.return_basis == "split_adjusted_price":
            if self.dividend_treatment != "excluded":
                raise ValueError(
                    "split_adjusted_price requires dividend_treatment='excluded'"
                )
        else:
            if self.dividend_treatment != "cash_received_no_reinvestment":
                raise ValueError(
                    "cash_dividend_inclusive requires an explicit cash dividend "
                    "treatment; supported policy is cash_received_no_reinvestment"
                )
            if self.corporate_action_evidence != "INDEPENDENTLY_VERIFIED":
                raise ValueError(
                    "dividend-inclusive targets require independently verified "
                    "corporate-action evidence"
                )


def validate_target_contract(contract: TargetContract) -> dict[str, str | int]:
    """Return a serializable declaration; does not authorize model training."""
    return {
        "return_basis": contract.return_basis,
        "horizon_sessions": contract.horizon_sessions,
        "decision_price_field": contract.decision_price_field,
        "entry_price_field": contract.entry_price_field,
        "exit_price_field": contract.exit_price_field,
        "label_end_field": contract.label_end_field,
        "dividend_treatment": contract.dividend_treatment,
        "corporate_action_evidence": contract.corporate_action_evidence,
        "training_authorized": 0,
        "label_generation_implemented": 0,
    }
