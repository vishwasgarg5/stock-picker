from __future__ import annotations

"""Phases 107-115: paper-trading evidence and release-readiness diagnostics.

All reports are analysis-only. No missing facts are inferred, no source ledgers
are rewritten, and no production ranking, risk, sizing, or promotion settings change.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAFE = {
    "diagnostic_only": True, "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
    "production_model_changed": False, "ranking_or_risk_changed": False,
    "risk_or_sizing_changed": False, "automatic_promotion": False,
    "missing_facts_inferred": False,
}
MIN_V2_TRADES = 100
MIN_V1_CHRONOLOGICAL_SESSIONS = 60
MIN_V2_SESSIONS = 17


def _read(name: str, **kwargs: Any) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _write(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = {**SAFE, **payload}
    (DATA / name).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def _buy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "signal" not in frame:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["signal"].astype(str).str.upper().str.strip().isin(["BUY", "LONG", "EXECUTED"])].copy()


def _fixed_share_source_evidence(buys: pd.DataFrame) -> dict[str, Any]:
    """Validate alternate fixed-share source fields without populating ledger columns.

    The pipeline's fixed-share path records quantity, position_value, planned_capital,
    and gross profit_loss, but does not model transaction costs. These facts are
    therefore reported separately and can never make the net-accounting gate pass.
    """
    required = [
        "quantity", "position_value", "planned_capital", "reference_price",
        "entry_price", "exit_price", "profit_loss", "return_pct",
    ]
    if not all(column in buys.columns for column in required):
        return {
            "status": "UNAVAILABLE",
            "candidate_rows": 0,
            "validated_rows": 0,
            "mismatch_rows": None,
            "missing_source_fields": [c for c in required if c not in buys.columns],
            "costs_modeled": False,
            "source_ledger_mutated": False,
        }
    numeric = buys[required].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1), index=buys.index
    )
    positive = (
        numeric["quantity"].gt(0)
        & numeric["position_value"].gt(0)
        & numeric["planned_capital"].gt(0)
        & numeric["reference_price"].gt(0)
        & numeric["entry_price"].gt(0)
        & numeric["exit_price"].gt(0)
    )
    candidates = finite & positive
    position_delta = numeric["quantity"] * numeric["entry_price"] - numeric["position_value"]
    gross_delta = numeric["quantity"] * (numeric["exit_price"] - numeric["entry_price"]) - numeric["profit_loss"]
    return_delta = (
        numeric["profit_loss"] / numeric["position_value"].replace(0, np.nan) * 100.0
        - numeric["return_pct"]
    )
    mismatch = (
        position_delta.abs().gt(0.05)
        | gross_delta.abs().gt(0.05)
        | return_delta.abs().gt(0.05)
    )
    split_metadata_present = (
        "pre_split_quantity" in buys.columns or "split_ratio" in buys.columns
    )
    invalid_split_metadata = pd.Series(False, index=buys.index)
    if split_metadata_present:
        if not {"pre_split_quantity", "split_ratio"}.issubset(buys.columns):
            invalid_split_metadata |= True
        else:
            pre_split = pd.to_numeric(buys["pre_split_quantity"], errors="coerce")
            ratio = pd.to_numeric(buys["split_ratio"], errors="coerce")
            split_metadata_finite = pd.Series(
                np.isfinite(pre_split.to_numpy(dtype=float))
                & np.isfinite(ratio.to_numpy(dtype=float)),
                index=buys.index,
            )
            invalid_split_metadata = (
                ~split_metadata_finite | pre_split.le(0) | ratio.le(0)
            )
            split_delta = pre_split * ratio - numeric["quantity"]
            mismatch |= split_delta.abs().gt(0.05)
            reference_delta = pre_split * numeric["reference_price"] - numeric["planned_capital"]
            mismatch |= reference_delta.abs().gt(0.05)
    candidates &= ~invalid_split_metadata
    valid = candidates & ~mismatch
    candidate_count = int(candidates.sum())
    mismatch_count = int((candidates & mismatch).sum())
    return {
        "status": "GROSS_SOURCE_EVIDENCE_VALIDATED" if candidate_count == len(buys) and candidate_count > 0 and mismatch_count == 0 else (
            "GROSS_SOURCE_EVIDENCE_PARTIAL" if valid.any() else "NO_VALID_SOURCE_EVIDENCE"
        ),
        "candidate_rows": candidate_count,
        "validated_rows": int(valid.sum()),
        "mismatch_rows": mismatch_count,
        "invalid_or_incomplete_source_rows": int((~candidates).sum()),
        "invalid_split_metadata_rows": int(invalid_split_metadata.sum()),
        "max_abs_position_value_delta": float(position_delta[candidates].abs().max()) if candidates.any() else None,
        "max_abs_gross_pnl_delta": float(gross_delta[candidates].abs().max()) if candidates.any() else None,
        "max_abs_return_delta_pct_points": float(return_delta[candidates].abs().max()) if candidates.any() else None,
        "source_semantics": "quantity and position_value are recorded source fields; profit_loss is gross P&L from quantity * (exit_price - entry_price)",
        "costs_modeled": False,
        "net_profit_loss_verified": False,
        "source_ledger_mutated": False,
        "policy": "Validate alternate source fields only; do not copy them into canonical accounting columns or assume transaction costs are zero.",
    }


def phase107_pnl_reconciliation(trades: pd.DataFrame) -> dict[str, Any]:
    """Reconcile recorded accounting and localize gaps without backfilling values."""
    buys = _buy_rows(trades)
    source_evidence = _fixed_share_source_evidence(buys)
    required = ["entry_price", "exit_price", "shares", "allocated_capital", "gross_profit_loss", "costs", "profit_loss", "return_pct"]
    missing_columns = [c for c in required if c not in buys.columns]
    if missing_columns:
        return _write("phase107_pnl_reconciliation_summary.json", {
            "status": "BLOCKED", "source": "data/paper_trades.csv", "buy_rows": len(buys),
            "missing_columns": missing_columns, "incomplete_rows": len(buys),
            "reconciliation_mismatch_rows": None,
            "fixed_share_source_evidence": source_evidence,
            "policy": "No historical values are backfilled or inferred.",
        })

    numeric = buys[required].apply(pd.to_numeric, errors="coerce")
    finite_values = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)),
        index=numeric.index, columns=numeric.columns,
    )
    invalid_non_finite_by_field = {
        c: int((numeric[c].notna() & ~finite_values[c]).sum()) for c in required
    }
    incomplete = numeric.isna().any(axis=1) | ~finite_values.all(axis=1)
    accounting_delta = numeric["gross_profit_loss"] - numeric["costs"] - numeric["profit_loss"]
    share_delta = numeric["shares"] * (numeric["exit_price"] - numeric["entry_price"]) - numeric["gross_profit_loss"]
    expected_return = numeric["profit_loss"] / numeric["allocated_capital"].replace(0, np.nan) * 100.0
    return_delta = expected_return - numeric["return_pct"]
    invalid_capital = numeric["allocated_capital"].le(0)
    mismatch = (
        (accounting_delta.abs() > 0.02)
        | (share_delta.abs() > 0.05)
        | (return_delta.abs() > 0.05)
        | invalid_capital
    )

    missing_by_field = {c: int(numeric[c].isna().sum()) for c in required}
    incomplete_by_signal = {}
    if "signal" in buys:
        labels = buys["signal"].fillna("MISSING").astype(str).str.upper().str.strip().replace("", "MISSING")
        incomplete_by_signal = {
            str(label): int((incomplete & labels.eq(label)).sum())
            for label in labels[incomplete & incomplete].unique()
        }
    session_detail = []
    if "target_date" in buys:
        dates = pd.to_datetime(buys["target_date"], errors="coerce").dt.normalize()
        detail = pd.DataFrame({"target_date": dates, "incomplete": incomplete.to_numpy()})
        detail = detail.dropna(subset=["target_date"]).groupby("target_date", as_index=False).agg(
            buy_rows=("incomplete", "size"), incomplete_rows=("incomplete", "sum")
        )
        session_detail = [
            {"target_date": str(row.target_date.date()), "buy_rows": int(row.buy_rows),
             "incomplete_rows": int(row.incomplete_rows)}
            for row in detail.itertuples(index=False) if row.incomplete_rows
        ]
    complete_dates = pd.to_datetime(
        buys.loc[~incomplete, "target_date"], errors="coerce"
    ).dropna() if "target_date" in buys else pd.Series(dtype="datetime64[ns]")
    incomplete_dates = pd.to_datetime(
        buys.loc[incomplete, "target_date"], errors="coerce"
    ).dropna() if "target_date" in buys else pd.Series(dtype="datetime64[ns]")
    # Classify record vintage only from explicit metadata, never from target-date age.
    if "ledger_schema_version" in buys:
        schema = buys["ledger_schema_version"].fillna("").astype(str).str.strip()
        versioned = schema.ne("")
    else:
        schema = pd.Series("", index=buys.index, dtype=str)
        versioned = pd.Series(False, index=buys.index)
    if "record_created_at_utc" in buys:
        created = pd.to_datetime(buys["record_created_at_utc"], errors="coerce", utc=True)
    else:
        created = pd.Series(pd.NaT, index=buys.index, dtype="datetime64[ns, UTC]")
    vintage = pd.Series("UNCLASSIFIED_NO_EXPLICIT_METADATA", index=buys.index, dtype=str)
    vintage.loc[versioned] = "SCHEMA_VERSIONED:" + schema.loc[versioned]
    vintage.loc[~versioned & created.notna()] = "CREATION_TIMESTAMP_ONLY"
    vintage_counts = {str(k): int(v) for k, v in vintage.value_counts(dropna=False).items()}
    incomplete_vintage_counts = {
        str(k): int(v) for k, v in vintage.loc[incomplete].value_counts(dropna=False).items()
    }

    status = "PASS" if len(buys) and not incomplete.any() and not mismatch.any() else ("INSUFFICIENT_DATA" if not len(buys) else "BLOCKED")
    return _write("phase107_pnl_reconciliation_summary.json", {
        "status": status, "source": "data/paper_trades.csv", "buy_rows": int(len(buys)),
        "missing_columns": [], "complete_rows": int((~incomplete).sum()),
        "incomplete_rows": int(incomplete.sum()),
        "missing_values_by_field": missing_by_field,
        "invalid_non_finite_values_by_field": invalid_non_finite_by_field,
        "invalid_non_finite_rows": int((~finite_values.all(axis=1)).sum()),
        "incomplete_rows_by_signal": incomplete_by_signal,
        "incomplete_rows_by_session": session_detail,
        "first_complete_session": str(complete_dates.min().date()) if len(complete_dates) else None,
        "last_complete_session": str(complete_dates.max().date()) if len(complete_dates) else None,
        "first_incomplete_session": str(incomplete_dates.min().date()) if len(incomplete_dates) else None,
        "last_incomplete_session": str(incomplete_dates.max().date()) if len(incomplete_dates) else None,
        "record_vintage_classification": "EXPLICIT_METADATA_WHEN_AVAILABLE; OTHERWISE_UNCLASSIFIED",
        "record_vintage_counts": vintage_counts,
        "incomplete_rows_by_record_vintage": incomplete_vintage_counts,
        "reconciliation_mismatch_rows": int((mismatch & ~incomplete).sum()),
        "accounting_identity": "gross_profit_loss - costs = profit_loss",
        "gross_pnl_identity": "shares * (exit_price - entry_price) = gross_profit_loss",
        "return_identity": "profit_loss / allocated_capital * 100 = return_pct",
        "max_abs_accounting_delta": float(accounting_delta[~incomplete].abs().max()) if (~incomplete).any() else None,
        "max_abs_gross_delta": float(share_delta[~incomplete].abs().max()) if (~incomplete).any() else None,
        "max_abs_return_delta_pct_points": float(return_delta[~incomplete].abs().max()) if (~incomplete).any() else None,
        "costs_subtracted_again": False,
        "fixed_share_source_evidence": source_evidence,
        "policy": "Uses recorded canonical values for the net ledger and separately validates alternate fixed-share source fields. No historical values are backfilled or inferred; alternate gross source evidence is not treated as proof of net accounting because costs are not modeled and it does not satisfy the net-accounting gate.",
    })

def phase108_loss_concentration(trades: pd.DataFrame) -> dict[str, Any]:
    buys = _buy_rows(trades)
    if buys.empty or "profit_loss" not in buys:
        return _write("phase108_loss_concentration_summary.json", {"status": "INSUFFICIENT_DATA", "trade_rows": len(buys), "loss_rows": 0})
    buys["profit_loss"] = pd.to_numeric(buys["profit_loss"], errors="coerce")
    buys = buys.dropna(subset=["profit_loss"])
    finite_pnl = np.isfinite(buys["profit_loss"].to_numpy(dtype=float))
    invalid_non_finite_rows = int((~finite_pnl).sum())
    if invalid_non_finite_rows:
        return _write("phase108_loss_concentration_summary.json", {
            "status": "BLOCKED_INVALID_METRICS", "trade_rows": int(len(buys)),
            "invalid_non_finite_rows": invalid_non_finite_rows,
            "gross_loss": None, "net_pnl": None, "group_reports": [],
            "note": "Non-finite P&L values detected; loss concentration was not scored.",
        })
    total_gross_loss = float(-buys.loc[buys.profit_loss < 0, "profit_loss"].sum())
    outputs = []
    for key, filename in [("symbol", "phase108_loss_by_symbol.csv"), ("target_date", "phase108_loss_by_session.csv"), ("exit_reason", "phase108_loss_by_exit_reason.csv")]:
        if key not in buys:
            continue
        x = buys.assign(_group=buys[key].fillna("NOT_RECORDED").astype(str).str.strip().replace("", "NOT_RECORDED"))
        grouped = x.groupby("_group", dropna=False).agg(trades=("profit_loss", "size"), net_pnl=("profit_loss", "sum"), losing_trades=("profit_loss", lambda s: int((s < 0).sum())), gross_loss=("profit_loss", lambda s: float(-s[s < 0].sum()))).reset_index().rename(columns={"_group": key})
        grouped["share_of_gross_loss_pct"] = grouped["gross_loss"] / total_gross_loss * 100 if total_gross_loss else 0.0
        grouped.sort_values(["gross_loss", "trades"], ascending=[False, False]).to_csv(DATA / filename, index=False)
        outputs.append(filename)
    return _write("phase108_loss_concentration_summary.json", {"status": "SCORED" if len(buys) >= 20 else "SMALL_SAMPLE", "trade_rows": int(len(buys)), "loss_rows": int((buys.profit_loss < 0).sum()), "gross_loss": total_gross_loss, "net_pnl": float(buys.profit_loss.sum()), "group_reports": outputs, "production_rules_changed": False})


def phase109_selection_funnel(v2: pd.DataFrame) -> dict[str, Any]:
    """Explain the observed V2 funnel and rejection subreasons without changing gates."""
    signal = v2.get("signal", pd.Series("", index=v2.index)).astype(str).str.upper().str.strip()
    skips = v2.loc[signal.eq("SKIP")]
    buys = v2.loc[signal.isin(["BUY", "LONG", "EXECUTED"])]
    reasons = skips.get("no_trade_reason", pd.Series("", index=skips.index)).fillna("").astype(str).str.strip()
    funnel = _read("phase32_v2_selection_funnel.csv")
    required = {"stage", "input_rows", "output_rows"}
    funnel_valid = bool(not funnel.empty and required.issubset(funnel.columns))
    funnel_stages = []
    funnel_validation_errors = []
    if funnel_valid:
        stage_names = funnel["stage"].fillna("").astype(str).str.strip()
        if stage_names.eq("").any():
            funnel_validation_errors.append("missing_stage_name")
        if stage_names.duplicated(keep=False).any():
            funnel_validation_errors.append("duplicate_stage_name")
        normalized_counts = {}
        for column in ("input_rows", "output_rows"):
            values = pd.to_numeric(funnel[column], errors="coerce")
            valid_counts = (
                values.notna()
                & np.isfinite(values.to_numpy(dtype=float))
                & values.ge(0)
                & values.mod(1).eq(0)
            )
            if not valid_counts.all():
                funnel_validation_errors.append(f"invalid_{column}")
            normalized_counts[column] = values
        output_exceeds_input = normalized_counts["output_rows"] > normalized_counts["input_rows"]
        if output_exceeds_input.any():
            funnel_validation_errors.append("output_rows_exceed_input_rows")
        if "rejected_rows" in funnel.columns:
            rejected = pd.to_numeric(funnel["rejected_rows"], errors="coerce")
            supplied = funnel["rejected_rows"].notna()
            valid_rejected = (
                ~supplied
                | (
                    rejected.notna()
                    & np.isfinite(rejected.fillna(0).to_numpy(dtype=float))
                    & rejected.ge(0)
                    & rejected.mod(1).eq(0)
                    & rejected.eq(normalized_counts["input_rows"] - normalized_counts["output_rows"])
                )
            )
            if not valid_rejected.all():
                funnel_validation_errors.append("invalid_rejected_rows")
            normalized_counts["unclassified_rows"] = (
                normalized_counts["input_rows"] - normalized_counts["output_rows"]
            ).where(~supplied, 0)
            # Keep the legacy numeric rejected_rows field for downstream
            # compatibility, but never interpret an absent value as a
            # confirmed zero-rejection count.
            normalized_counts["rejected_rows"] = rejected.fillna(0)
        else:
            normalized_counts["rejected_rows"] = (
                normalized_counts["input_rows"] - normalized_counts["output_rows"]
            )
            normalized_counts["unclassified_rows"] = 0
        if not funnel_validation_errors:
            normalized = funnel.copy()
            for column, values in normalized_counts.items():
                normalized[column] = values.astype(int)
            normalized["stage"] = stage_names
            funnel_stages = normalized.to_dict(orient="records")
        else:
            funnel_valid = False
    phase4_reasons = {}
    if "phase4_reject_reason" in skips:
        phase4 = skips["phase4_reject_reason"].fillna("").astype(str).str.strip()
        phase4_reasons = {
            str(k or "NOT_RECORDED"): int(v)
            for k, v in phase4[phase4.ne("")].value_counts().items()
        }
    # Prefer the persisted stage funnel for gate counts. The V2 decision ledger
    # does not persist phase2_selected, so row-level cross-tabs would otherwise
    # misleadingly classify every decision as a missing flag.
    stage_by_name = {
        str(row.get("stage", "")): row for row in funnel_stages
    }
    required_gate_stages = {
        "confidence_percentile_gate", "phase2_selected_gate", "phase4_eligibility_gate"
    }
    missing_required_gate_stages = sorted(required_gate_stages.difference(stage_by_name))
    required_gate_stages_present = bool(funnel_valid and not missing_required_gate_stages)
    confidence_stage = stage_by_name.get("confidence_percentile_gate", {})
    phase2_stage = stage_by_name.get("phase2_selected_gate", {})
    phase4_stage = stage_by_name.get("phase4_eligibility_gate", {})
    confidence_pct = pd.to_numeric(v2.get("confidence_pct", pd.Series(np.nan, index=v2.index)), errors="coerce")
    phase2_flag = pd.to_numeric(v2.get("phase2_selected", pd.Series(np.nan, index=v2.index)), errors="coerce")
    has_phase2_flag = "phase2_selected" in v2.columns
    gate_diagnostics = {
        "source": "persisted_selection_funnel",
        "shadow_confidence_percentile_threshold": 0.50,
        "confidence_gate_input_rows": int(confidence_stage.get("input_rows", 0)),
        "confidence_gate_output_rows": int(confidence_stage.get("output_rows", 0)),
        "confidence_gate_rejected_rows": int(confidence_stage.get("rejected_rows", 0)),
        "phase2_gate_input_rows": int(phase2_stage.get("input_rows", 0)),
        "phase2_gate_output_rows": int(phase2_stage.get("output_rows", 0)),
        "phase2_gate_rejected_rows": int(phase2_stage.get("rejected_rows", 0)),
        "phase4_eligibility_input_rows": int(phase4_stage.get("input_rows", 0)),
        "phase4_eligibility_output_rows": int(phase4_stage.get("output_rows", 0)),
        "phase4_eligibility_rejected_rows": int(phase4_stage.get("rejected_rows", 0)),
        "confidence_pct_non_null_rows": int(confidence_pct.notna().sum()),
        "confidence_pct_missing_rows": int(confidence_pct.isna().sum()),
        "phase2_flag_persisted_in_decision_ledger": bool(has_phase2_flag),
        "phase2_flag_missing_or_invalid_rows": int(phase2_flag.isna().sum()) if has_phase2_flag else None,
        "phase2_selected_and_confidence_below_threshold_rows": int((phase2_flag.eq(1) & confidence_pct.lt(0.50)).sum()) if has_phase2_flag else None,
        "phase2_selected_and_confidence_at_least_threshold_rows": int((phase2_flag.eq(1) & confidence_pct.ge(0.50)).sum()) if has_phase2_flag else None,
    }
    if len(buys):
        zero_buy_bottleneck = "BUY_ROWS_PRESENT"
    elif not required_gate_stages_present:
        zero_buy_bottleneck = "INCOMPLETE_FUNNEL_STAGE_DIAGNOSTICS"
    elif int(confidence_stage.get("output_rows", 0)) == 0:
        zero_buy_bottleneck = "CONFIDENCE_GATE_ZERO_OUTPUT"
    elif int(phase2_stage.get("output_rows", 0)) == 0:
        zero_buy_bottleneck = "PHASE2_SELECTED_GATE_ZERO_OUTPUT"
    elif int(phase4_stage.get("output_rows", 0)) == 0:
        zero_buy_bottleneck = "PHASE4_ELIGIBILITY_GATE_ZERO_OUTPUT"
    elif int(stage_by_name.get("atr_sanity_gate", {}).get("output_rows", 0)) == 0:
        zero_buy_bottleneck = "ATR_SANITY_GATE_ZERO_OUTPUT"
    elif int(stage_by_name.get("daily_trade_cap", {}).get("output_rows", 0)) == 0:
        zero_buy_bottleneck = "DAILY_TRADE_CAP_ZERO_OUTPUT"
    else:
        zero_buy_bottleneck = "NO_ZERO_OUTPUT_GATE_IDENTIFIED_CHECK_DECISION_MAPPING"

    reason_by_phase2 = {}
    if "no_trade_reason" in skips and has_phase2_flag:
        tmp = pd.DataFrame({"reason": skips["no_trade_reason"].fillna("").astype(str).str.strip(),
                            "phase2_selected": phase2_flag.reindex(skips.index)})
        tmp["reason"] = tmp["reason"].replace("", "NOT_RECORDED")
        reason_by_phase2 = {
            str(reason): {str(flag): int(count) for flag, count in group["phase2_selected"].value_counts(dropna=False).items()}
            for reason, group in tmp.groupby("reason", dropna=False)
        }


    confidence_stats = {}
    for column in ("confidence_pct", "confidence_v3", "confidence_score"):
        if column in v2:
            values = pd.to_numeric(v2[column], errors="coerce").dropna()
            confidence_stats[column] = {
                "non_null_rows": int(len(values)),
                "min": float(values.min()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
    return _write("phase109_selection_funnel_summary.json", {
        "status": "SCORED" if len(v2) else "INSUFFICIENT_DATA",
        "v2_rows": int(len(v2)), "buy_rows": int(len(buys)), "skip_rows": int(len(skips)),
        "skip_rows_without_reason": int(reasons.eq("").sum()),
        "skip_reason_counts": {str(k): int(v) for k, v in reasons.replace("", "NOT_RECORDED").value_counts().items()},
        "phase4_reject_reason_counts": phase4_reasons,
        "zero_buy_bottleneck": zero_buy_bottleneck,
        "confidence_column_diagnostics": confidence_stats,
        "gate_overlap_diagnostics": gate_diagnostics,
        "skip_reasons_by_phase2_flag": reason_by_phase2,
        "skip_reason_cross_tab_status": "AVAILABLE" if has_phase2_flag else "UNAVAILABLE_PHASE2_FLAG_NOT_PERSISTED; USE_FUNNEL_STAGE_COUNTS",
        "existing_funnel_valid": funnel_valid,
        "funnel_validation_errors": funnel_validation_errors,
        "required_gate_stages_present": required_gate_stages_present,
        "missing_required_gate_stages": missing_required_gate_stages,
        "funnel_stage_rows": int(len(funnel)) if funnel_valid else 0,
        "funnel_stages": funnel_stages,
        "interpretation": "Rejection counts describe the recorded shadow run; they do not prove a gate is miscalibrated. Review source-stage counts and confidence distributions before considering any threshold changes.",
        "production_rules_changed": False,
    })

def phase110_forecast_accuracy() -> dict[str, Any]:
    # Score only persisted realized evaluations. Joining the current forecast
    # file to OHLCV here can silently discard unmatched forecasts and mixes
    # next-session predictions with already-realized observations.
    evaluations = _read("evaluations.csv")
    predicted = {"open": "predicted_open", "high": "predicted_high", "low": "predicted_low", "close": "predicted_close"}
    actual_ohlc = [f"actual_{field}" for field in ("open", "high", "low", "close")]
    required = {"symbol", "target_date", "base_close", "predicted_close", *actual_ohlc}
    missing = sorted(required.difference(evaluations.columns))
    if evaluations.empty or missing:
        return _write("phase110_forecast_accuracy_summary.json", {
            "status": "INSUFFICIENT_DATA", "matched_rows": 0, "metrics": {},
            "missing_evaluation_columns": missing,
            "source": "data/evaluations.csv",
            "note": "Phase 110 requires persisted realized evaluations; it does not fall back to current predictions or inner-join away unmatched forecasts.",
        })
    merged = evaluations.copy()
    merged["symbol"] = merged["symbol"].astype(str).str.upper().str.strip()
    merged["target_date"] = pd.to_datetime(merged["target_date"], errors="coerce").dt.normalize()
    invalid_keys = int((merged["symbol"].eq("") | merged["symbol"].eq("NAN") | merged["target_date"].isna()).sum())
    if invalid_keys:
        return _write("phase110_forecast_accuracy_summary.json", {
            "status": "BLOCKED_INVALID_KEYS", "matched_rows": 0,
            "invalid_evaluation_key_rows": invalid_keys, "metrics": {},
            "close_directional_accuracy_pct": None,
            "note": "Missing or unparseable symbol/date keys detected in realized evaluations; no forecast metrics were scored.",
        })
    duplicates = int(merged.duplicated(["symbol", "target_date"], keep=False).sum())
    if duplicates:
        return _write("phase110_forecast_accuracy_summary.json", {
            "status": "BLOCKED_AMBIGUOUS_KEYS", "matched_rows": 0,
            "duplicate_evaluation_key_rows": duplicates, "metrics": {},
            "close_directional_accuracy_pct": None,
            "note": "Duplicate symbol/target_date keys detected in realized evaluations; ambiguous rows were not scored.",
        })
    available_predictions = {field: column for field, column in predicted.items() if column in merged.columns}
    numeric_columns = list(available_predictions.values()) + actual_ohlc + ["base_close"]
    numeric = merged[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(numeric_values).all(axis=1)
    positive = numeric.gt(0).all(axis=1).to_numpy(dtype=bool)
    ohlc = numeric[actual_ohlc]
    internally_consistent = (
        (ohlc["actual_high"] >= ohlc[["actual_open", "actual_low", "actual_close"]].max(axis=1))
        & (ohlc["actual_low"] <= ohlc[["actual_open", "actual_high", "actual_close"]].min(axis=1))
    ).to_numpy(dtype=bool)
    invalid_ohlc = ~internally_consistent
    if invalid_ohlc.any():
        return _write("phase110_forecast_accuracy_summary.json", {
            "status": "BLOCKED_INVALID_OHLC", "matched_rows": int(len(merged)),
            "invalid_actual_ohlc_rows": int(invalid_ohlc.sum()), "metrics": {},
            "close_directional_accuracy_pct": None,
            "note": "Realized actual OHLC values are internally inconsistent; no metrics were scored.",
        })
    invalid_rows = ~(finite & positive)
    if invalid_rows.any():
        return _write("phase110_forecast_accuracy_summary.json", {
            "status": "BLOCKED_INVALID_METRICS", "matched_rows": int(len(merged)),
            "invalid_metric_rows": int(invalid_rows.sum()), "metrics": {},
            "close_directional_accuracy_pct": None,
            "note": "Realized evaluation rows contain missing, non-finite, or non-positive forecast/actual prices or base-close values; no metrics were scored.",
        })
    metrics = {}
    for field, pcol in available_predictions.items():
        p = numeric[pcol]
        a = numeric[f"actual_{field}"]
        mae = float((p - a).abs().mean())
        bias = float((p - a).mean())
        mape_pct = float(((p - a).abs() / a.abs()).mean() * 100.0)
        if not np.isfinite(np.asarray([mae, bias, mape_pct], dtype=float)).all():
            return _write("phase110_forecast_accuracy_summary.json", {
                "status": "BLOCKED_INVALID_METRICS", "matched_rows": int(len(merged)),
                "invalid_metric_rows": int(len(merged)), "metrics": {},
                "close_directional_accuracy_pct": None,
                "note": "Derived forecast error metrics overflowed; no accuracy metrics were scored.",
            })
        metrics[field] = {"n": int(len(merged)), "mae": mae, "bias": bias, "mape_pct": mape_pct}
    pc, ac, base = numeric["predicted_close"], numeric["actual_close"], numeric["base_close"]
    directional = float((((pc > base) == (ac > base)).mean()) * 100.0)
    return _write("phase110_forecast_accuracy_summary.json", {
        "status": "SCORED" if len(merged) else "INSUFFICIENT_DATA",
        "source": "data/evaluations.csv",
        "matched_rows": int(len(merged)), "metrics": metrics,
        "close_directional_accuracy_pct": directional,
        "chronological_split_used_for_this_summary": False,
        "note": "Descriptive accuracy over persisted realized forecast evaluations; all evaluation rows are validated and scored. Use Phase 111 for chronological replay.",
    })

def _daily_metrics(frame: pd.DataFrame, date_col: str, pnl_col: str) -> dict[str, Any]:
    empty = {
        "rows": 0, "sessions": 0, "net_pnl": None,
        "max_drawdown_pct_from_100k_start": None, "win_rate_pct": None,
        "profit_factor": None, "average_session_pnl": None,
        "best_session_pnl": None, "worst_session_pnl": None,
        "first_session": None, "last_session": None,
        "chronological_order": True,
        "costs_source": "recorded net profit_loss; costs are not subtracted again",
    }
    if frame.empty or date_col not in frame or pnl_col not in frame:
        return empty
    x = frame.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce").dt.normalize()
    x[pnl_col] = pd.to_numeric(x[pnl_col], errors="coerce")
    valid_dates = x[date_col].notna()
    finite_pnl = pd.Series(
        np.isfinite(x[pnl_col].to_numpy(dtype=float)), index=x.index
    )
    invalid_date_rows = int((~valid_dates).sum())
    invalid_pnl_rows = int((~x[pnl_col].notna() | ~finite_pnl).sum())
    invalid_metric_rows = int((~valid_dates | ~x[pnl_col].notna() | ~finite_pnl).sum())
    if invalid_metric_rows:
        return {
            **empty,
            "rows": int(len(x)),
            "invalid_metric_rows": invalid_metric_rows,
            "invalid_date_rows": invalid_date_rows,
            "invalid_pnl_rows": invalid_pnl_rows,
            "valid": False,
        }
    x = x.sort_values(date_col)
    if x.empty:
        return empty
    pnl_values = x[pnl_col].to_numpy(dtype=float)
    if not np.isfinite(pnl_values).all():
        return {**empty, "invalid_metric_rows": int((~np.isfinite(pnl_values)).sum()), "valid": False}
    daily = x.groupby(date_col)[pnl_col].sum().sort_index()
    starting_capital = 100000.0
    # Include starting equity in the peak series so a first-session loss is
    # measured as drawdown from the initial capital rather than a zero drawdown.
    equity = pd.concat([
        pd.Series([starting_capital], index=[daily.index.min() - pd.Timedelta(days=1)]),
        starting_capital + daily.cumsum(),
    ]).sort_index()
    peak = equity.cummax()
    drawdown = ((equity / peak) - 1.0) * 100.0
    trade_pnl = pd.to_numeric(x[pnl_col], errors="coerce").dropna()
    gains = float(trade_pnl[trade_pnl > 0].sum())
    losses = float(-trade_pnl[trade_pnl < 0].sum())
    return {
        "rows": int(len(x)),
        "sessions": int(daily.size),
        "invalid_metric_rows": 0,
        "valid": True,
        "net_pnl": float(daily.sum()),
        "max_drawdown_pct_from_100k_start": float(drawdown.min()) if len(drawdown) else None,
        "win_rate_pct": float((trade_pnl > 0).mean() * 100.0) if len(trade_pnl) else None,
        "positive_session_rate_pct": float((daily > 0).mean() * 100.0),
        "profit_factor": gains / losses if losses > 0 else (None if gains == 0 else "UNBOUNDED_NO_LOSSES"),
        "average_session_pnl": float(daily.mean()),
        "best_session_pnl": float(daily.max()),
        "worst_session_pnl": float(daily.min()),
        "first_session": str(daily.index.min().date()),
        "last_session": str(daily.index.max().date()),
        "chronological_order": True,
        "costs_source": "recorded net profit_loss; costs are not subtracted again",
    }


def phase111_chronological_replay(v1: pd.DataFrame, v2: pd.DataFrame) -> dict[str, Any]:
    v1m = _daily_metrics(_buy_rows(v1), "target_date", "profit_loss")
    v2m = _daily_metrics(_buy_rows(v2), "target_date", "profit_loss")
    enough = (
        v1m.get("valid", True) and v2m.get("valid", True)
        and v1m["sessions"] >= MIN_V1_CHRONOLOGICAL_SESSIONS
        and v2m["sessions"] >= MIN_V2_SESSIONS
    )
    return _write("phase111_chronological_replay_summary.json", {"status": "SCORED" if enough else "INSUFFICIENT_EVIDENCE", "v1": v1m, "v2": v2m, "minimum_v1_sessions_for_evidence": MIN_V1_CHRONOLOGICAL_SESSIONS, "minimum_v2_sessions_for_evidence": MIN_V2_SESSIONS, "caveat": "Chronological aggregation of recorded simulated outcomes, not a claim of executable live returns."})


def phase112_market_intelligence() -> dict[str, Any]:
    """Validate market-intelligence dates against the latest available OHLCV session."""
    frame = _read("index_intelligence_summary.csv")
    history = _read("ohlcv.csv", usecols=lambda c: c in {"date"})
    required = {"index", "as_of"}
    if frame.empty or not required.issubset(frame.columns) or history.empty or "date" not in history:
        return _write("phase112_market_intelligence_summary.json", {
            "status": "INSUFFICIENT_DATA",
            "source": "data/index_intelligence_summary.csv",
            "rows": int(len(frame)),
            "latest_record": frame.tail(1).to_dict(orient="records")[0] if not frame.empty else None,
            "missing_columns": sorted(required.difference(frame.columns)),
            "reference_ohlcv_session": None,
            "invalid_as_of_rows": 0,
            "duplicate_index_rows": 0,
            "stale_index_rows": None,
            "news_context_source_present": (DATA / "phase26_news_impact.csv").exists(),
            "note": "A market-intelligence row is not considered current without valid index dates and an OHLCV session reference.",
        })
    dates = pd.to_datetime(frame["as_of"], errors="coerce").dt.normalize()
    names = frame["index"].fillna("").astype(str).str.strip().str.upper()
    invalid = int((dates.isna() | names.eq("")).sum())
    duplicate = int(names[names.ne("")].duplicated(keep=False).sum())
    ohlcv_dates = pd.to_datetime(history["date"], errors="coerce").dropna().dt.normalize()
    reference = ohlcv_dates.max() if len(ohlcv_dates) else pd.NaT
    stale = int((dates.notna() & dates.lt(reference)).sum()) if pd.notna(reference) else None
    future = int((dates.notna() & dates.gt(reference)).sum()) if pd.notna(reference) else None
    valid = invalid == 0 and duplicate == 0 and pd.notna(reference) and stale == 0 and future == 0
    status = "SCORED" if valid else "BLOCKED_INVALID_DATA" if invalid or duplicate or future else "STALE_DATA" if stale else "INSUFFICIENT_DATA"
    latest = frame.tail(1).to_dict(orient="records")
    return _write("phase112_market_intelligence_summary.json", {
        "status": status,
        "source": "data/index_intelligence_summary.csv",
        "rows": int(len(frame)),
        "latest_record": latest[0] if latest else None,
        "reference_ohlcv_session": str(reference.date()) if pd.notna(reference) else None,
        "latest_index_session": str(dates.max().date()) if dates.notna().any() else None,
        "invalid_as_of_rows": invalid,
        "duplicate_index_rows": duplicate,
        "stale_index_rows": stale,
        "future_index_rows": future,
        "news_context_source_present": (DATA / "phase26_news_impact.csv").exists(),
        "note": "Market intelligence is current only when every index has one valid dated record aligned to the latest available OHLCV session; no arbitrary calendar-day freshness window is assumed.",
    })

def phase113_sustained_evidence(v1: pd.DataFrame, v2: pd.DataFrame) -> dict[str, Any]:
    def evidence(frame: pd.DataFrame) -> dict[str, Any]:
        buys = _buy_rows(frame)
        raw_dates = buys.get("target_date", pd.Series(dtype=str))
        parsed_dates = pd.to_datetime(raw_dates, errors="coerce").dt.normalize()
        invalid_date_rows = int(parsed_dates.isna().sum())
        dates = parsed_dates.dropna()
        return {
            "executed_rows": int(len(buys)),
            "invalid_session_date_rows": invalid_date_rows,
            "valid": invalid_date_rows == 0,
            "distinct_sessions": int(dates.nunique()),
            "first_session": str(dates.min().date()) if len(dates) else None,
            "last_session": str(dates.max().date()) if len(dates) else None,
            "minimum_trade_count_met": bool(len(buys) >= MIN_V2_TRADES),
            "minimum_session_count_met": bool(dates.nunique() >= MIN_V2_SESSIONS),
        }
    v1e, v2e = evidence(v1), evidence(v2)
    enough = (
        v1e["valid"] and v2e["valid"]
        and v2e["minimum_trade_count_met"] and v2e["minimum_session_count_met"]
    )
    return _write("phase113_sustained_evidence_summary.json", {"status": "SCORED" if enough else "INSUFFICIENT_EVIDENCE", "v1": v1e, "v2": v2e, "minimum_v2_trades": MIN_V2_TRADES, "minimum_distinct_sessions": MIN_V2_SESSIONS, "note": "Minimum counts are necessary but not sufficient; invalid execution dates block sustained-evidence scoring; market-regime coverage and uncertainty still require review."})


def phase114_promotion_gate(
    recon: dict[str, Any],
    funnel: dict[str, Any],
    forecast: dict[str, Any],
    replay: dict[str, Any],
    sustained: dict[str, Any],
    exit_attribution: dict[str, Any] | None = None,
    loss_concentration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exit_attribution = exit_attribution or {}
    loss_concentration = loss_concentration or {}
    gates = {
        "pnl_reconciliation_pass": recon.get("status") == "PASS",
        "loss_concentration_evidence_available": loss_concentration.get("status") == "SCORED",
        "selection_reasons_complete": funnel.get("status") == "SCORED" and funnel.get("skip_rows_without_reason", 0) == 0,
        "selection_funnel_stage_coverage": funnel.get("existing_funnel_valid") is True and funnel.get("required_gate_stages_present") is True,
        "exit_reason_attribution_complete": (
            exit_attribution.get("status") == "SCORED"
            and exit_attribution.get("unknown_or_unrecorded_reason_rows", 1) == 0
        ),
        "forecast_metrics_available": forecast.get("status") == "SCORED",
        "chronological_evidence_sufficient": replay.get("status") == "SCORED",
        "sustained_v2_sample_sufficient": sustained.get("status") == "SCORED",
    }
    failed = [name for name, passed in gates.items() if not passed]
    return _write("phase114_promotion_readiness_summary.json", {"status": "INSUFFICIENT_EVIDENCE" if failed else "REVIEW_REQUIRED", "gates": gates, "unmet_gates": failed, "v1_remains_production_champion": True, "v2_remains_shadow_only": True, "automatic_promotion": False, "production_changes_authorized": False, "note": "Even a clean diagnostic gate requires explicit human review; it never promotes V2 automatically."})


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    v1, v2 = _read("paper_trades.csv"), _read("paper_trades_v2.csv")
    recon = phase107_pnl_reconciliation(v1)
    loss = phase108_loss_concentration(v1)
    funnel = phase109_selection_funnel(v2)
    forecast = phase110_forecast_accuracy()
    replay = phase111_chronological_replay(v1, v2)
    market = phase112_market_intelligence()
    sustained = phase113_sustained_evidence(v1, v2)
    try:
        exit_attribution = json.loads(
            (DATA / "phase105_exit_attribution_summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, UnicodeDecodeError):
        exit_attribution = {}
    gate = phase114_promotion_gate(
        recon, funnel, forecast, replay, sustained, exit_attribution, loss
    )
    required_reports = [
        "phase107_pnl_reconciliation_summary.json",
        "phase108_loss_concentration_summary.json",
        "phase109_selection_funnel_summary.json",
        "phase110_forecast_accuracy_summary.json",
        "phase111_chronological_replay_summary.json",
        "phase112_market_intelligence_summary.json",
        "phase113_sustained_evidence_summary.json",
        "phase114_promotion_readiness_summary.json",
    ]
    reports_generated = all((DATA / name).is_file() for name in required_reports)
    monitor = _write("phase115_shadow_monitoring_summary.json", {
        "status": "GENERATED" if reports_generated else "BLOCKED",
        "diagnostic_reports_generated": reports_generated,
        "status_semantics": "GENERATED means all diagnostic reports were written; it does not mean the evidence passed, the model is profitable, or V2 is ready for promotion.",
        "phase_statuses": {
            "107_pnl": recon.get("status"), "108_losses": loss.get("status"),
            "109_funnel": funnel.get("status"), "110_forecast": forecast.get("status"),
            "111_replay": replay.get("status"), "112_market": market.get("status"),
            "113_sustained": sustained.get("status"), "114_gate": gate.get("status"),
        },
        "production_changes_authorized": False,
        "v1_champion_v2_shadow_only": True,
    })
    print(json.dumps(monitor, indent=2, default=str))
    return monitor


if __name__ == "__main__":
    run()
