from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "final_model_validation.csv"

MIN_OOS_SESSIONS = 24
MIN_RANKING_SESSIONS = 12


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_final_validation() -> pd.DataFrame:
    wf = _read("walk_forward_evaluations.csv")
    sel = _read("model_selection.csv")
    rank = _read("ranking_validation_horizons.csv")
    costs = _read("ranking_validation_costs.csv")
    stability = _read("ranking_validation_stability.csv")
    sector = _read("ranking_validation_sector_risk.csv")

    sessions = int(wf["target_date"].nunique()) if not wf.empty and "target_date" in wf else 0
    model_mape = float(pd.to_numeric(wf.get("close_abs_pct_error"), errors="coerce").mean()) if not wf.empty else np.nan
    baseline_mape = float(pd.to_numeric(wf.get("baseline_close_abs_pct_error"), errors="coerce").mean()) if not wf.empty else np.nan
    model_vs_baseline = bool(np.isfinite(model_mape) and np.isfinite(baseline_mape) and model_mape <= baseline_mape)

    rank_sessions = int(rank["prediction_date"].nunique()) if not rank.empty and "prediction_date" in rank else 0
    top10 = rank[(rank.get("group") == "TOP10") & (rank.get("horizon_sessions") == 5)] if not rank.empty else pd.DataFrame()
    rank_lift = float(pd.to_numeric(top10.get("vs_11_20_mean_return_pct"), errors="coerce").mean()) if not top10.empty else np.nan
    ranking_evidence = bool(np.isfinite(rank_lift) and rank_lift > 0)

    net = costs[(costs.get("group") == "TOP10") & (costs.get("cost_bps_per_side") == 10.0)] if not costs.empty else pd.DataFrame()
    net_return = float(pd.to_numeric(net.get("net_return_pct"), errors="coerce").mean()) if not net.empty else np.nan
    cost_evidence = bool(np.isfinite(net_return))

    stability_ok = True
    if not stability.empty and "top10_overlap_pct" in stability:
        stability_ok = bool(pd.to_numeric(stability["top10_overlap_pct"], errors="coerce").notna().any())

    sector_ok = True
    if not sector.empty and "concentration_breach" in sector:
        sector_ok = not bool(sector["concentration_breach"].fillna(False).astype(bool).any())

    gate_checks = {
        "oos_sessions": sessions >= MIN_OOS_SESSIONS,
        "model_vs_baseline": model_vs_baseline,
        "ranking_sessions": rank_sessions >= MIN_RANKING_SESSIONS,
        "ranking_5d_lift_positive": ranking_evidence,
        "transaction_cost_evidence": cost_evidence,
        "stability_evidence": stability_ok,
        "sector_concentration": sector_ok,
    }
    passed = all(gate_checks.values())

    row = {
        "as_of": pd.Timestamp.now().normalize(),
        "oos_sessions": sessions,
        "ranking_sessions": rank_sessions,
        "model_close_mape_pct": model_mape * 100 if np.isfinite(model_mape) else np.nan,
        "baseline_close_mape_pct": baseline_mape * 100 if np.isfinite(baseline_mape) else np.nan,
        "model_vs_baseline_passed": model_vs_baseline,
        "top10_5d_mean_lift_vs_11_20_pct": rank_lift,
        "top10_10bps_mean_net_return_pct": net_return,
        "stability_evidence": stability_ok,
        "sector_concentration_passed": sector_ok,
        **{f"gate_{k}": v for k, v in gate_checks.items()},
        "promotion_gate_passed": passed,
        "decision": "PROMOTE" if passed else "HOLD",
        "decision_note": (
            "All final validation gates passed."
            if passed else
            "Retain current production model; at least one independent OOS/risk gate is not yet satisfied."
        ),
    }
    out = pd.DataFrame([row])
    out.to_csv(OUTPUT, index=False)
    return out


if __name__ == "__main__":
    print(run_final_validation().to_string(index=False))
