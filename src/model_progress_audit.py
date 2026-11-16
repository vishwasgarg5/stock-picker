"""Evidence-based, failure-isolated progress audit for the Stock Picker repository.

This is an inventory/diagnostic tool. File presence is never treated as proof of
model quality or production readiness; evidence statuses are surfaced verbatim.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "Universe management": "src/universe.py",
    "Core forecasting pipeline": "src/pipeline.py",
    "Market calendar": "src/market_calendar.py",
    "Listing-date master": "src/listing_master.py",
    "Historical coverage audit": "src/historical_coverage_audit.py",
    "Historical gap recovery": "src/repair_historical_gaps.py",
    "Index OHLCV and constituents": "src/index_constituent_intelligence.py",
    "Market/index regime intelligence": "src/index_intelligence.py",
    "News context": "src/news_context.py",
    "Ranking model": "src/ranking_model.py",
    "Directional model": "src/directional_model.py",
    "Confidence model": "src/confidence.py",
    "Trade-quality model": "src/trade_quality_model.py",
    "Risk management": "src/risk_management.py",
    "Paper trading V1": "src/paper_trading.py",
    "Paper trading V2": "src/paper_trading_v2.py",
    "Realized outcomes": "src/outcomes.py",
    "Model governance": "src/model_governance.py",
    "System health": "src/system_health.py",
    "Performance audit": "src/performance_audit.py",
}

ARTIFACTS = {
    "NSE universe": ("data/universe.csv", {"symbol"}),
    "Historical OHLCV": ("data/ohlcv.csv", {"date", "symbol", "open", "high", "low", "close", "volume"}),
    "Predictions": ("data/predictions.csv", {"symbol"}),
    "Prediction evaluations": ("data/evaluations.csv", set()),
    "Candidate history": ("data/prediction_candidates_history.csv", set()),
    "Paper trades V1": ("data/paper_trades.csv", set()),
    "Paper trades V2": ("data/paper_trades_v2.csv", set()),
    "NSE market calendar": ("data/nse_market_calendar.json", None),
    "Historical coverage summary": ("data/historical_coverage_audit.json", None),
    "Phase 107 P&L reconciliation": ("data/phase107_pnl_reconciliation_summary.json", None),
    "Phase 114 promotion readiness": ("data/phase114_promotion_readiness_summary.json", None),
    "Model governance": ("data/model_governance.json", None),
    "Index OHLCV": ("data/index_ohlcv.csv", set()),
    "Index constituent history": ("data/index_constituents_history.csv", set()),
    "Index constituent features": ("data/index_constituent_features.csv", set()),
    "Index intelligence audit": ("data/index_constituent_intelligence_audit.json", None),
    "Weekend data-quality audit": ("data/weekend_data_quality_audit.json", None),
    "Listing master": ("data/nse_listing_master.csv", set()),
}

PHASE_EVIDENCE = [
    "data/phase20_accuracy_summary.json", "data/phase21_26_summary.json",
    "data/phase27_data_quality_summary.json", "data/phase28_30_summary.json",
    "data/phase31_summary.json", "data/phase41_43_summary.json",
    "data/phase48_52_summary.json", "data/phase58_62_summary.json",
    "data/phase63_72_readiness_summary.json", "data/phase73_76_diagnostics.json",
    "data/phase77_80_diagnostics.json", "data/phase81_84_diagnostics.json",
    "data/phase85_88_diagnostics.json", "data/phase89_92_diagnostics.json",
    "data/phase93_96_diagnostics.json", "data/phase97_forecast_quality_summary.json",
    "data/phase98_101_quality_summary.json", "data/phase103_forecast_evidence_summary.json",
    "data/phase107_pnl_reconciliation_summary.json", "data/phase110_forecast_accuracy_summary.json",
    "data/phase111_chronological_replay_summary.json", "data/phase112_market_intelligence_summary.json",
    "data/phase113_sustained_evidence_summary.json", "data/phase114_promotion_readiness_summary.json",
    "data/phase115_shadow_monitoring_summary.json", "data/phase120_forecast_stability_summary.json",
    "data/phase122_challenger_decision.json", "data/phase123_magnitude_calibration_decision.json",
    "data/phase124_feature_signal_decision.json", "data/phase125_feature_ablation_decision.json",
    "data/phase126_objective_target_decision.json",
]

def add(rows, group, name, status, detail, evidence=""):
    rows.append({"group": group, "check": name, "status": status, "detail": str(detail), "evidence": evidence})

def inspect_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, (dict, list)):
        raise ValueError("top-level JSON must be an object or array")
    return value

def run_audit(output_dir: Path):
    rows = []
    for name, rel in MODULES.items():
        exists = (ROOT / rel).is_file()
        add(rows, "Implementation structure", name, "PASS" if exists else "FAIL",
            "Source module exists" if exists else "Required source module is missing", rel)

    for name, (rel, required_columns) in ARTIFACTS.items():
        path = ROOT / rel
        if not path.is_file():
            add(rows, "Data and evidence artifacts", name, "BLOCKED",
                "Artifact has not been produced or is not present; dependent validation cannot be confirmed", rel)
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = inspect_json(path)
                status = payload.get("status") if isinstance(payload, dict) else None
                detail = f"Valid JSON; reported status={status or 'not supplied'}"
                if status and str(status).upper() not in {"PASS", "OK", "HEALTHY", "COMPLETE", "SUFFICIENT"}:
                    detail += " (evidence exists but its own status is not a clean pass)"
                    result = "REVIEW"
                else:
                    result = "PASS"
            else:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                missing = sorted(required_columns - set(header)) if required_columns else []
                if missing:
                    result, detail = "FAIL", f"Missing required columns: {missing}"
                else:
                    result, detail = "PASS", f"Readable CSV; {len(header)} columns; header validated"
            add(rows, "Data and evidence artifacts", name, result, detail, rel)
        except Exception as exc:
            add(rows, "Data and evidence artifacts", name, "FAIL", f"Cannot parse artifact: {type(exc).__name__}: {exc}", rel)

    present = [p for p in PHASE_EVIDENCE if (ROOT / p).is_file()]
    missing = [p for p in PHASE_EVIDENCE if not (ROOT / p).is_file()]
    add(rows, "Roadmap evidence coverage", "Phase evidence summaries",
        "PASS" if not missing else "REVIEW",
        f"{len(present)}/{len(PHASE_EVIDENCE)} expected phase evidence summaries present; "
        f"missing: {', '.join(missing) if missing else 'none'}",
        "data/phase*_summary.json")

    for rel in ["README.md", ".github/workflows/daily.yml",
                ".github/workflows/weekend-data-audit.yml",
                ".github/workflows/pipeline-integrity-ci.yml"]:
        exists = (ROOT / rel).is_file()
        add(rows, "Operations and documentation", rel, "PASS" if exists else "FAIL",
            "Present" if exists else "Missing", rel)

    # Report important evidence gates as evidence, not as a binary code-health result.
    for rel, label in [
        ("data/phase114_promotion_readiness_summary.json", "Production promotion gate"),
        ("data/phase107_pnl_reconciliation_summary.json", "Paper-trade P&L reconciliation"),
        ("data/historical_coverage_audit.json", "Historical data coverage"),
    ]:
        path = ROOT / rel
        if not path.exists():
            add(rows, "Model readiness evidence", label, "BLOCKED", "No report available", rel)
            continue
        try:
            payload = inspect_json(path)
            status = str(payload.get("status", "STATUS_NOT_REPORTED")).upper()
            detail = f"Current evidence report status: {status}"
            if status in {"PASS", "SUFFICIENT", "READY", "COMPLETE"}:
                result = "PASS"
            elif status in {"FAIL", "BLOCKED", "INSUFFICIENT_EVIDENCE"}:
                result = "BLOCKED"
            else:
                result = "REVIEW"
            unmet = payload.get("unmet_gates", [])
            if unmet:
                detail += f"; unmet gates: {', '.join(map(str, unmet))}"
            add(rows, "Model readiness evidence", label, result, detail, rel)
        except Exception as exc:
            add(rows, "Model readiness evidence", label, "FAIL", f"Invalid report: {exc}", rel)

    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ["PASS", "FAIL", "BLOCKED", "REVIEW"]}
    total = len(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "repository structure and persisted evidence; not a profitability certification",
        "counts": counts, "total_checks": total, "checks": rows,
    }
    (output_dir / "model_progress_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Stock Picker — Model Progress Audit",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "> This audit separates code structure, data artifacts and readiness evidence. "
        "A present file is not proof of model skill, and BLOCKED/REVIEW are real findings, not workflow failures.",
        "",
        "## Summary",
        "",
        f"- Total checks: **{total}**",
        f"- PASS: **{counts['PASS']}**",
        f"- FAIL: **{counts['FAIL']}**",
        f"- BLOCKED: **{counts['BLOCKED']}**",
        f"- REVIEW: **{counts['REVIEW']}**",
        "",
        "## Detailed checks",
        "",
        "| Group | Check | Status | Detail | Evidence |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        clean = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
        lines.append("| " + " | ".join(clean(r[k]) for k in ["group", "check", "status", "detail", "evidence"]) + " |")
    (output_dir / "model_progress_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"Full report: {output_dir / 'model_progress_audit.md'}")
    return 0 if counts["FAIL"] == 0 else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="audit/model_progress")
    args = parser.parse_args()
    raise SystemExit(run_audit(ROOT / args.output_dir))
