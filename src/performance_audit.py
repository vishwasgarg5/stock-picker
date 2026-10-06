from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "performance_audit.csv"
SUMMARY = DATA / "performance_audit_summary.csv"


def read(name, date_col=None):
    p = DATA / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=[date_col] if date_col else None)
    except Exception:
        return pd.DataFrame()


def _metric_block(e):
    if e.empty:
        return {}
    out = {"evaluation_rows": len(e)}
    if "target_date" not in e.columns:
        return out

    x = e.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce")
    for c in [
        "close_abs_pct_error", "open_abs_pct_error", "high_abs_pct_error",
        "low_abs_pct_error", "baseline_close_abs_pct_error", "close_direction_correct"
    ]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    for c in ["close_abs_pct_error", "open_abs_pct_error", "high_abs_pct_error",
              "low_abs_pct_error", "baseline_close_abs_pct_error"]:
        if c in x:
            v = x[c].dropna()
            if not v.empty:
                out[c] = float(v.mean() * 100)

    if "close_direction_correct" in x:
        v = x["close_direction_correct"].dropna()
        if not v.empty:
            out["direction_accuracy_pct"] = float(v.mean() * 100)

    sessions = (
        x.dropna(subset=["target_date"])
        .groupby("target_date", as_index=False)
        [["close_abs_pct_error", "baseline_close_abs_pct_error"]]
        .mean()
        .sort_values("target_date")
    )
    for n in (5, 10, 20):
        w = sessions.tail(n)
        if len(w) == n:
            out[f"recent_{n}_close_mape_pct"] = float(w["close_abs_pct_error"].mean() * 100)
            out[f"recent_{n}_baseline_close_mape_pct"] = float(w["baseline_close_abs_pct_error"].mean() * 100)
            out[f"recent_{n}_vs_baseline_improvement_pct"] = float(
                (1 - w["close_abs_pct_error"].mean() / max(w["baseline_close_abs_pct_error"].mean(), 1e-12)) * 100
            )
    return out


def run_audit():
    e = read("evaluations.csv", "target_date")
    v1 = read("portfolio_daily.csv", "target_date")
    v2 = read("portfolio_v2_daily.csv", "target_date")
    conf = read("confidence_validation_summary.csv", "as_of")
    rank = read("ranking_model_validation_summary.csv")
    ab = read("strategy_ab_comparison.csv", "as_of")

    rows = []
    m = _metric_block(e)
    for k, v in m.items():
        rows.append({"area": "OHLC/model", "metric": k, "value": v})

    if not v1.empty and "daily_profit_loss" in v1:
        rows += [
            {"area": "V1", "metric": "sessions", "value": len(v1)},
            {"area": "V1", "metric": "net_pnl", "value": pd.to_numeric(v1.daily_profit_loss, errors="coerce").sum()},
            {"area": "V1", "metric": "max_drawdown_pct", "value": pd.to_numeric(v1.get("drawdown_pct", pd.Series(dtype=float)), errors="coerce").min()},
        ]
    if not v2.empty and "daily_profit_loss" in v2:
        rows += [
            {"area": "V2", "metric": "sessions", "value": len(v2)},
            {"area": "V2", "metric": "net_pnl", "value": pd.to_numeric(v2.daily_profit_loss, errors="coerce").sum()},
            {"area": "V2", "metric": "max_drawdown_pct", "value": pd.to_numeric(v2.get("drawdown_pct", pd.Series(dtype=float)), errors="coerce").min()},
            {"area": "V2", "metric": "trades", "value": pd.to_numeric(v2.get("trades", pd.Series(dtype=float)), errors="coerce").sum()},
        ]
    if not conf.empty:
        r = conf.iloc[-1]
        for c in ["rows", "sessions", "top_confidence_mape_pct", "bottom_confidence_mape_pct",
                  "top_bottom_mape_improvement_pct", "top_confidence_direction_pct",
                  "bottom_confidence_direction_pct", "confidence_promotion_evidence"]:
            if c in r:
                rows.append({"area": "Confidence", "metric": c, "value": r[c]})
    if not rank.empty:
        r = rank.iloc[-1]
        for c in [
            "sessions", "rows", "mean_top10_return_lift_pct", "positive_rate_lift_pct",
            "promotion_evidence", "production_ready", "status",
            "recent_5_lift_pct", "recent_10_lift_pct", "recent_20_lift_pct",
            "recent_5_positive_lift_pct", "recent_10_positive_lift_pct", "recent_20_positive_lift_pct"
        ]:
            if c in r:
                rows.append({"area": "Ranking challenger", "metric": c, "value": r[c]})
    if not ab.empty:
        r = ab.iloc[-1]
        for c in ["common_sessions", "v2_trades", "pnl_lift", "return_lift_pct",
                  "session_win_rate_pct", "production_strategy", "status"]:
            if c in r:
                rows.append({"area": "Strategy A/B", "metric": c, "value": r[c]})

    audit = pd.DataFrame(rows)
    audit["as_of"] = pd.Timestamp.now().normalize()
    audit.to_csv(OUT, index=False)

    findings = []
    if m.get("close_abs_pct_error") is not None and m.get("baseline_close_abs_pct_error") is not None:
        findings.append(
            "MODEL_BEATS_BASELINE"
            if m["close_abs_pct_error"] < m["baseline_close_abs_pct_error"]
            else "MODEL_NOT_BEATING_BASELINE"
        )
    if not conf.empty and str(conf.iloc[-1].get("confidence_promotion_evidence", "False")).lower() == "true":
        findings.append("CONFIDENCE_READY")
    else:
        findings.append("CONFIDENCE_NOT_PROMOTED")

    if not rank.empty and bool(rank.iloc[-1].get("production_ready", False)):
        findings.append("RANKING_PRODUCTION_READY")
    elif not rank.empty and str(rank.iloc[-1].get("promotion_evidence", "False")).lower() == "true":
        findings.append("RANKING_VALIDATED_HOLD")
    else:
        findings.append("RANKING_NOT_PROMOTED")

    if not ab.empty:
        findings.append("V2_CURRENTLY_" + str(ab.iloc[-1].get("status", "UNKNOWN")).upper())

    summary = pd.DataFrame([{
        "as_of": pd.Timestamp.now().normalize(),
        "findings": " | ".join(findings),
        "audit_rows": len(audit)
    }])
    summary.to_csv(SUMMARY, index=False)
    print(summary.to_string(index=False))
    return audit


if __name__ == "__main__":
    run_audit()
