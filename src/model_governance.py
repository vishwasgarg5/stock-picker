from __future__ import annotations

"""Connected model/strategy governance checks.

This module implements the remaining validation layers without changing the
production champion by itself.  Every result is persisted for the governor and
dashboard; promotion remains gated by independent matched-session evidence.
"""

from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DIRS = [ROOT / "src", ROOT / ".github" / "workflows"]
TARGET_THRESHOLDS = (0.0010, 0.0015, 0.0025, 0.0050)
BOOTSTRAPS = 5000


def _read(name: str, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    p = DATA / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=list(dates) or None)
    except Exception:
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()


def _target_diagnostics(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty or not {"date", "symbol", "close"}.issubset(hist.columns):
        return pd.DataFrame()
    x = hist.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.sort_values(["symbol", "date"])
    x["next_return"] = x.groupby("symbol")["close"].shift(-1) / x["close"] - 1.0
    x = x.dropna(subset=["next_return"])
    rows = []
    for t in TARGET_THRESHOLDS:
        eligible = x[x["next_return"].abs() >= t]
        if eligible.empty:
            continue
        pos = float((eligible["next_return"] > 0).mean())
        rows.append({
            "threshold_pct": t * 100.0,
            "eligible_rows": int(len(eligible)),
            "coverage_pct": float(len(eligible) / max(len(x), 1) * 100.0),
            "positive_rate_pct": pos * 100.0,
            "class_balance_distance_pct": abs(pos - 0.5) * 100.0,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["selected_threshold"] = np.isclose(out["threshold_pct"], 0.15)
    return out


def _regime_report(hist: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    if evaluations.empty or "target_date" not in evaluations.columns:
        return pd.DataFrame()
    e = evaluations.copy()
    e["target_date"] = pd.to_datetime(e["target_date"], errors="coerce").dt.normalize()
    if "close_direction_correct" not in e.columns:
        return pd.DataFrame()
    e["direction_correct"] = pd.to_numeric(e["close_direction_correct"], errors="coerce")
    e = e.dropna(subset=["target_date", "direction_correct"])
    if e.empty:
        return pd.DataFrame()
    if "regime" not in e.columns:
        h = hist.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
        h["close"] = pd.to_numeric(h["close"], errors="coerce")
        daily = h.dropna(subset=["date","close"]).groupby("date")["close"].median().sort_index()
        s20, s50 = daily.rolling(20).mean(), daily.rolling(50).mean()
        r20 = daily / daily.shift(20) - 1.0
        regimes = pd.Series("NEUTRAL", index=daily.index)
        regimes[(daily > s20) & (s20 > s50) & (r20 >= 0.03)] = "BULL"
        regimes[(daily < s20) & (s20 < s50) & (r20 <= -0.03)] = "BEAR"
        e = e.merge(regimes.rename("regime"), left_on="target_date", right_index=True, how="left")
        e["regime"] = e["regime"].fillna("NEUTRAL")
    return e.groupby("regime", dropna=False).agg(
        rows=("direction_correct","size"),
        direction_accuracy_pct=("direction_correct", lambda s: float(s.mean()*100.0)),
    ).reset_index()


def _stock_stability(evaluations: pd.DataFrame) -> pd.DataFrame:
    if evaluations.empty or "symbol" not in evaluations.columns or "close_direction_correct" not in evaluations.columns:
        return pd.DataFrame()
    x = evaluations.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["direction_correct"] = pd.to_numeric(x["close_direction_correct"], errors="coerce")
    x["target_date"] = pd.to_datetime(x.get("target_date"), errors="coerce").dt.normalize()
    x = x.dropna(subset=["symbol","direction_correct","target_date"])
    x = x.sort_values(["symbol", "target_date"], kind="mergesort")
    if x.empty:
        return pd.DataFrame()
    latest = x["target_date"].max()
    g = x.groupby("symbol").agg(
        rows=("direction_correct","size"),
        accuracy_pct=("direction_correct", lambda s: float(s.mean()*100.0)),
        recent_accuracy_pct=("direction_correct", lambda s: float(s.tail(10).mean()*100.0)),
    ).reset_index()
    g["stability_gap_pct"] = (g["accuracy_pct"] - g["recent_accuracy_pct"]).abs()
    g["stable"] = (g["rows"] >= 10) & (g["accuracy_pct"] >= 50.0) & (g["recent_accuracy_pct"] >= 50.0)
    g["as_of"] = latest
    return g.sort_values(["stable","accuracy_pct"], ascending=[True,False])


def _feature_stability(hist: pd.DataFrame) -> pd.DataFrame:
    try:
        from .pipeline import FEATURE_COLUMNS, features
        x = features(hist)
        if x.empty:
            return pd.DataFrame()
        x["target_return"] = x.groupby("symbol")["close"].shift(-1) / x["close"] - 1.0
        x = x.dropna(subset=["target_return"])
        windows = []
        dates = sorted(pd.to_datetime(x["date"]).dropna().unique())
        if len(dates) < 60:
            return pd.DataFrame()
        cuts = [dates[-20:], dates[-40:-20], dates[-60:-40]]
        for i, d in enumerate(cuts, 1):
            part = x[x["date"].isin(d)]
            for f in FEATURE_COLUMNS:
                a = pd.to_numeric(part[f], errors="coerce")
                b = pd.to_numeric(part["target_return"], errors="coerce")
                corr = a.corr(b)
                windows.append({"window": i, "feature": f, "correlation": float(corr) if pd.notna(corr) else np.nan})
        out = pd.DataFrame(windows)
        g = out.groupby("feature").agg(
            windows_valid=("correlation","count"),
            mean_correlation=("correlation","mean"),
            min_correlation=("correlation","min"),
            max_correlation=("correlation","max"),
        ).reset_index()
        g["sign_consistent"] = (g["min_correlation"] >= 0) | (g["max_correlation"] <= 0)
        g["stable_signal"] = (g["windows_valid"] == 3) & g["sign_consistent"] & (g["mean_correlation"].abs() >= 0.02)
        return g.sort_values("mean_correlation", key=lambda s: s.abs(), ascending=False)
    except Exception:
        return pd.DataFrame()


def _bootstrap_ab() -> dict:
    x = _read("strategy_ab_sessions.csv")
    if x.empty or "return_difference_pct" not in x.columns:
        return {"sessions": 0, "bootstrap_ci_low_pct": -np.inf, "bootstrap_ci_high_pct": np.inf, "bootstrap_mean_pct": 0.0}
    d = pd.to_numeric(x["return_difference_pct"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(d) < 2:
        return {"sessions": int(len(d)), "bootstrap_ci_low_pct": -np.inf, "bootstrap_ci_high_pct": np.inf, "bootstrap_mean_pct": float(d.mean()) if len(d) else 0.0}
    rng = np.random.default_rng(20261007)
    samples = rng.choice(d, size=(BOOTSTRAPS, len(d)), replace=True).mean(axis=1)
    return {
        "sessions": int(len(d)),
        "bootstrap_mean_pct": float(d.mean()),
        "bootstrap_ci_low_pct": float(np.quantile(samples, 0.025)),
        "bootstrap_ci_high_pct": float(np.quantile(samples, 0.975)),
    }




def _tail_risk() -> dict:
    trades = _read("paper_trades_v2.csv")
    if trades.empty or "profit_loss" not in trades.columns:
        return {"v2_trades": 0, "worst_trade_pnl": np.nan, "consecutive_losses": 0, "profit_factor": np.nan}
    if "signal" in trades.columns:
        trades = trades[trades["signal"].astype(str).str.upper().eq("BUY")].copy()
    if "target_date" in trades.columns:
        trades["target_date"] = pd.to_datetime(trades["target_date"], errors="coerce")
        trades = trades.sort_values(["target_date", "symbol"] if "symbol" in trades.columns else ["target_date"], kind="mergesort")
    pnl = pd.to_numeric(trades["profit_loss"], errors="coerce").dropna()
    if pnl.empty:
        return {"v2_trades": 0, "worst_trade_pnl": np.nan, "consecutive_losses": 0, "profit_factor": np.nan}
    loss_runs = 0
    max_loss_run = 0
    for value in pnl.to_numpy():
        if value < 0:
            loss_runs += 1
            max_loss_run = max(max_loss_run, loss_runs)
        else:
            loss_runs = 0
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return {
        "v2_trades": int(len(pnl)),
        "worst_trade_pnl": float(pnl.min()),
        "consecutive_losses": int(max_loss_run),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
    }

def _drift(hist: pd.DataFrame) -> dict:
    try:
        from .pipeline import FEATURE_COLUMNS, features
        x = features(hist)
        dates = sorted(pd.to_datetime(x["date"]).dropna().unique())
        if len(dates) < 40:
            return {"status":"collecting","feature_drift_count":0}
        recent_dates, ref_dates = dates[-20:], dates[-60:-20]
        recent, ref = x[x["date"].isin(recent_dates)], x[x["date"].isin(ref_dates)]
        drift = []
        for f in FEATURE_COLUMNS:
            a = pd.to_numeric(recent[f], errors="coerce").dropna()
            b = pd.to_numeric(ref[f], errors="coerce").dropna()
            if a.empty or b.empty:
                continue
            scale = max(abs(float(b.median())), 1e-9)
            drift.append({"feature":f,"relative_median_shift":float((a.median()-b.median())/scale)})
        d = pd.DataFrame(drift)
        count = int((d["relative_median_shift"].abs() >= 0.50).sum()) if not d.empty else 0
        return {"status":"PASS" if count <= 4 else "WARNING","feature_drift_count":count,"max_relative_median_shift":float(d["relative_median_shift"].abs().max()) if not d.empty else 0.0}
    except Exception:
        return {"status":"WARNING","feature_drift_count":0,"max_relative_median_shift":0.0}




def _model_drift() -> dict:
    h = _read("performance_history.csv")
    if h.empty or "direction_accuracy_pct" not in h.columns:
        return {"status": "collecting", "recent_direction_accuracy_pct": np.nan, "prior_direction_accuracy_pct": np.nan}
    if "target_date" in h.columns:
        h = h.assign(_target_date=pd.to_datetime(h["target_date"], errors="coerce"))
        h = h.sort_values("_target_date", kind="mergesort")
    elif "date" in h.columns:
        h = h.assign(_target_date=pd.to_datetime(h["date"], errors="coerce"))
        h = h.sort_values("_target_date", kind="mergesort")
    x = pd.to_numeric(h["direction_accuracy_pct"], errors="coerce").dropna()
    if len(x) < 10:
        return {"status": "collecting", "recent_direction_accuracy_pct": float(x.mean()) if len(x) else np.nan, "prior_direction_accuracy_pct": np.nan}
    recent = float(x.tail(5).mean())
    prior = float(x.iloc[-10:-5].mean())
    return {
        "status": "PASS" if recent >= prior - 5.0 else "WARNING",
        "recent_direction_accuracy_pct": recent,
        "prior_direction_accuracy_pct": prior,
        "direction_accuracy_change_pct": recent - prior,
    }

def _manifest() -> dict:
    files = []
    for root in SOURCE_DIRS:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")) if root.name == "src" else sorted(root.rglob("*.yml")):
            files.append(str(p.relative_to(ROOT)))
    hashes = {}
    for rel in files:
        hashes[rel] = sha256((ROOT / rel).read_bytes()).hexdigest()
    return {"files": len(hashes), "sha256": hashes, "manifest_version":"governance_v1"}


def _safety() -> dict:
    required = ["data/ohlcv.csv","data/predictions.csv"]
    missing = [p for p in required if not (ROOT / p).exists()]
    failures = list(missing)
    p = _read("predictions.csv")
    if not p.empty:
        for c in ["symbol","rank","base_close","predicted_open","predicted_high","predicted_low","predicted_close"]:
            if c not in p.columns:
                failures.append("prediction_missing:"+c)
        if {"symbol","rank"}.issubset(p.columns):
            date_col = next((c for c in ["target_date", "date", "prediction_date"] if c in p.columns), None)
            if date_col is not None:
                key = pd.to_datetime(p[date_col], errors="coerce").dt.normalize().astype(str)
                sym = p["symbol"].astype(str).str.upper().str.strip()
                rank = pd.to_numeric(p["rank"], errors="coerce")
                if pd.DataFrame({"date": key, "symbol": sym}).duplicated().any():
                    failures.append("duplicate_prediction_date_symbol")
                if pd.DataFrame({"date": key, "rank": rank}).duplicated().any():
                    failures.append("duplicate_prediction_date_rank")
    state = {}
    state_path = ROOT / "data/strategy_state.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {"production_strategy": "V1"}
    except Exception:
        failures.append("invalid_strategy_state")
    if str(state.get("production_strategy","V1")).upper() not in {"V1","V2"}:
        failures.append("invalid_production_strategy")
    return {"status":"PASS" if not failures else "FAIL","failures":failures,"v1_safe_default":str(state.get("production_strategy","V1")).upper()=="V1"}


def run_governance() -> dict:
    DATA.mkdir(exist_ok=True)
    hist = _read("ohlcv.csv")
    evals = _read("evaluations.csv")
    targets = _target_diagnostics(hist)
    regimes = _regime_report(hist, evals)
    stocks = _stock_stability(evals)
    features = _feature_stability(hist)
    bootstrap = _bootstrap_ab()
    drift = _drift(hist)
    tail = _tail_risk()
    model_drift = _model_drift()
    safety = _safety()
    confidence = _read("confidence_validation_summary.csv")
    selection = _read("phase2_performance_summary.csv")
    ranking_validation = _read("ranking_model_validation_summary.csv")
    v1_trades = _read("paper_trades.csv")
    manifest = _manifest()
    strategy_state_path = DATA / "strategy_state.json"
    phase114_path = DATA / "phase114_promotion_readiness_summary.json"
    bootstrap_valid = (
        int(bootstrap.get("sessions", 0)) >= 2
        and np.isfinite(float(bootstrap.get("bootstrap_ci_low_pct", np.nan)))
        and np.isfinite(float(bootstrap.get("bootstrap_ci_high_pct", np.nan)))
    )
    steps = {
        "step_01_walk_forward": not evals.empty and "target_date" in evals.columns,
        "step_02_target_optimization": not targets.empty,
        "step_03_probability_calibration": not confidence.empty and "confidence_promotion_evidence" in confidence.columns,
        "step_04_regime_validation": not regimes.empty,
        "step_05_stock_stability": not stocks.empty,
        "step_06_feature_stability": not features.empty,
        "step_07_ranking_optimization": not ranking_validation.empty and str(ranking_validation.iloc[-1].get("diagnostic_ready", False)).lower() == "true",
        "step_08_matched_session_ledger": not v1_trades.empty and {"symbol", "target_date", "signal"}.issubset(v1_trades.columns),
        "step_09_v2_evidence_gate": phase114_path.exists(),
        "step_10_confidence_buckets": not confidence.empty and {"confidence_bucket", "rows"}.issubset(confidence.columns),
        "step_11_tail_risk": tail["v2_trades"] > 0,
        "step_12_bootstrap": bootstrap_valid,
        "step_13_statistical_promotion": bootstrap_valid and float(bootstrap["bootstrap_ci_low_pct"]) > 0,
        "step_14_promotion_hysteresis": strategy_state_path.exists(),
        "step_15_automatic_rollback": (DATA / "rollback_summary.json").exists(),
        "step_16_reproducibility_manifest": manifest["files"] > 0,
        "step_17_data_drift": drift["status"] == "PASS",
        "step_18_model_drift": model_drift["status"] == "PASS",
        "step_19_no_trade_safety": safety["status"] == "PASS",
        "step_20_consolidated_dashboard": (DATA / "performance_history.csv").exists(),
    }
    report = {
        "as_of": pd.Timestamp.now(tz="UTC").isoformat(),
        "target_threshold_current_pct": 0.15,
        "target_threshold_candidates_pct": ",".join(str(x*100) for x in TARGET_THRESHOLDS),
        "target_threshold_best_coverage_pct": float(targets.loc[targets["coverage_pct"].idxmax(),"threshold_pct"]) if not targets.empty else np.nan,
        "regime_rows": int(len(regimes)),
        "stock_rows": int(len(stocks)),
        "stock_stability_status": "SCORED" if not stocks.empty else "INSUFFICIENT_DATA",
        "unstable_stocks": int((~stocks["stable"]).sum()) if not stocks.empty and "stable" in stocks else None,
        "stable_features": int(features["stable_signal"].sum()) if not features.empty and "stable_signal" in features else None,
        "feature_stability_status": "SCORED" if not features.empty else "INSUFFICIENT_DATA",
        "feature_count": int(len(features)),
        "bootstrap_sessions": bootstrap["sessions"],
        "bootstrap_mean_return_lift_pct": bootstrap["bootstrap_mean_pct"],
        "bootstrap_ci_low_pct": bootstrap["bootstrap_ci_low_pct"],
        "bootstrap_ci_high_pct": bootstrap["bootstrap_ci_high_pct"],
        "drift_status": drift["status"],
        "model_drift_status": model_drift["status"],
        "recent_direction_accuracy_pct": model_drift.get("recent_direction_accuracy_pct", np.nan),
        "prior_direction_accuracy_pct": model_drift.get("prior_direction_accuracy_pct", np.nan),
        "direction_accuracy_change_pct": model_drift.get("direction_accuracy_change_pct", np.nan),
        "feature_drift_count": drift["feature_drift_count"],
        "max_feature_median_shift": drift["max_relative_median_shift"],
        "confidence_evidence": bool(str(confidence.iloc[-1].get("confidence_promotion_evidence",False)).lower()=="true") if not confidence.empty else False,
        "phase2_enabled": bool(str(selection.iloc[-1].get("production_enabled",False)).lower()=="true") if not selection.empty else False,
        "safety_status": safety["status"],
        "safety_failures": "|".join(safety["failures"]),
        "v1_safe_default": safety["v1_safe_default"],
        "tail_v2_trades": tail["v2_trades"],
        "worst_v2_trade_pnl": tail["worst_trade_pnl"],
        "max_consecutive_v2_losses": tail["consecutive_losses"],
        "v2_profit_factor": tail["profit_factor"],
        "governance_version": "v1.0",
        **steps,
        "governance_steps_with_evidence": int(sum(bool(value) for value in steps.values())),
        "governance_steps_total": len(steps),
        "governance_status": "EVIDENCE_AVAILABLE_REVIEW_REQUIRED" if all(steps.values()) else "INSUFFICIENT_EVIDENCE",
        "step_semantics": "A true step means its source evidence is present; it does not mean the model passed that evidence or is promotable.",
        "step_evidence_status": {name: ("AVAILABLE" if value else "MISSING_OR_NOT_PASSED") for name, value in steps.items()},
        "v1_remains_production_champion": True,
        "v2_remains_shadow_only": True,
        "automatic_promotion": False,
        "production_changes_authorized": False,
    }
    pd.DataFrame([report]).to_csv(DATA/"model_governance_summary.csv",index=False)
    targets.to_csv(DATA/"target_threshold_diagnostics.csv",index=False)
    regimes.to_csv(DATA/"regime_performance.csv",index=False)
    stocks.to_csv(DATA/"stock_stability.csv",index=False)
    features.to_csv(DATA/"feature_stability.csv",index=False)
    (DATA/"model_manifest.json").write_text(json.dumps(manifest,indent=2))
    (DATA/"model_governance.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    if safety["status"] != "PASS":
        raise RuntimeError("Governance safety gate failed: " + "; ".join(safety["failures"]))
    return report


if __name__ == "__main__":
    run_governance()
