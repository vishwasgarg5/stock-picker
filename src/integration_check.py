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
        "src/phase2_optimizer.py", "src/phase2_dashboard.py",
        "src/directional_model.py",
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
        if workflow.count("src.phase2_optimizer") < 2:
            failures.append("phase2_optimizer_refresh_after_v2_missing")
        if "src.phase2_dashboard" not in workflow:
            failures.append("phase2_dashboard_not_wired_to_workflow")

    pipeline = (ROOT / "src/pipeline.py").read_text(encoding="utf-8")
    v2 = (ROOT / "src/paper_trading_v2.py").read_text(encoding="utf-8")
    governor = (ROOT / "src/strategy_governor.py").read_text(encoding="utf-8")
    portfolio = (ROOT / "src/portfolio_comparison.py").read_text(encoding="utf-8")
    phase2 = (ROOT / "src/phase2_optimizer.py").read_text(encoding="utf-8")

    links = {
        "confidence_to_v2": "confidence_validation_summary.csv" in v2 and "_confidence_gate" in v2,
        "shadow_safety_to_v2": "MAX_SHADOW_ATR_PCT" in v2,
        "quality_to_pipeline": "latest_trade_quality_scores" in pipeline,
        "ranking_to_pipeline": "latest_rank_scores" in pipeline,
        "v2_to_governor": "portfolio_v2_daily.csv" in governor,
        "governor_to_portfolio_report": "strategy_state.json" in portfolio,
        "governor_v1_safe_default": 'production_strategy": "V1"' in governor,
        "confidence_to_governor": "confidence_validation_summary.csv" in governor,
        "phase2_shadow_optimizer": "production_enabled" in phase2,
        "phase2_directional_engine": "direction_score_v3" in phase2,
        "phase2_repeat_loss_control": "repeat_loss_penalty" in phase2,
        "phase2_regime_engine": "_regime" in phase2,
        "phase2_shadow_return_measurement": "_shadow_selection_metrics" in phase2,
        "directional_challenger_module": "src/directional_model.py" in required,
        "directional_challenger_pipeline": "directional_model" in pipeline,
    }
    failures.extend(f"link_broken:{name}" for name, ok in links.items() if not ok)

    generated = [
        "data/confidence_validation_summary.csv",
        "data/strategy_state.json",
        "data/strategy_ab_comparison.csv",
        "data/performance_audit_summary.csv",
        "data/portfolio_comparison_summary.csv",
        "data/phase2_performance_summary.csv",
        "data/phase2_optimized_candidates.csv",
        "data/phase2_dashboard.csv",
        "data/phase2_dashboard.json",
        "data/directional_model_validation_summary.csv",
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
