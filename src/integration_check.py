from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORKFLOW = ROOT / ".github" / "workflows" / "daily.yml"

def main() -> int:
    failures = []
    required = [
        "src/confidence_validation.py", "src/paper_trading_v2.py",
        "src/risk_management.py", "src/trade_quality_model.py",
        "src/ranking_model.py", "src/strategy_governor.py",
        "src/performance_audit.py", "src/portfolio_comparison.py",
        "data/predictions.csv", "data/ohlcv.csv",
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            failures.append(f"missing:{rel}")

    if WORKFLOW.exists():
        workflow = WORKFLOW.read_text(encoding="utf-8")
        if "push:" in workflow or "pull_request:" in workflow:
            failures.append("workflow_has_unwanted_push_or_pull_request_trigger")
        if "schedule:" not in workflow or "45 9 * * 1-5" not in workflow:
            failures.append("workflow_schedule_is_not_15_15_IST_weekdays")
        if "workflow_dispatch:" not in workflow or "reuse_stored_data" not in workflow:
            failures.append("workflow_dispatch_reuse_input_missing")
        if "concurrency:" not in workflow or "stock-picker-daily" not in workflow:
            failures.append("workflow_concurrency_guard_missing")
    else:
        failures.append("missing:.github/workflows/daily.yml")

    pipeline = (ROOT / "src/pipeline.py").read_text(encoding="utf-8")
    v2 = (ROOT / "src/paper_trading_v2.py").read_text(encoding="utf-8")
    risk = (ROOT / "src/risk_management.py").read_text(encoding="utf-8")
    governor = (ROOT / "src/strategy_governor.py").read_text(encoding="utf-8")
    portfolio = (ROOT / "src/portfolio_comparison.py").read_text(encoding="utf-8")

    links = {
        "confidence_to_v2": "confidence_validation_summary.csv" in v2 and "_confidence_gate" in v2,
        "risk_to_v2": "apply_risk_gate" in v2,
        "quality_to_pipeline": "latest_trade_quality_scores" in pipeline,
        "ranking_to_pipeline": "latest_rank_scores" in pipeline,
        "v2_to_governor": "portfolio_v2_daily.csv" in governor,
        "governor_to_portfolio_report": "strategy_state.json" in portfolio,
        "governor_v1_safe_default": 'production_strategy": "V1"' in governor,
        "confidence_to_governor": "confidence_validation_summary.csv" in governor,
    }
    failures.extend(f"link_broken:{name}" for name, ok in links.items() if not ok)

    generated = [
        "data/confidence_validation_summary.csv",
        "data/strategy_state.json",
        "data/strategy_ab_comparison.csv",
        "data/performance_audit_summary.csv",
        "data/portfolio_comparison_summary.csv",
    ]
    failures.extend(f"missing_generated_report:{p}" for p in generated if not (ROOT / p).exists())

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "as_of": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "integration_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
