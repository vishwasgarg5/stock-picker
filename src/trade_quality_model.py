from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_FILE = ROOT / "models" / "trade_quality_challenger.joblib"
SUMMARY_FILE = DATA / "trade_quality_validation_summary.csv"
VALIDATION_FILE = DATA / "trade_quality_validation.csv"
PREDICTIONS_FILE = DATA / "predictions.csv"
TRADES_FILE = DATA / "paper_trades.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"

MIN_TRADES = 50
MIN_SESSIONS = 10
MIN_VALIDATION_TRADES = 20
FEATURES = [
    "rank", "score", "confidence_score", "prediction_spread",
    "expected_return_pct", "predicted_range_pct",
    "recent_symbol_return_pct", "recent_symbol_win_rate",
    "recent_symbol_error_pct", "rank_percentile",
]
VERSION = "trade_quality_challenger_v1"


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _base_features(pred: pd.DataFrame, trades: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    p = pred.copy()
    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    for c in ["rank", "score", "confidence_score", "prediction_spread", "base_close",
              "predicted_high", "predicted_low", "predicted_close"]:
        if c in p:
            p[c] = _num(p[c])

    t = trades.copy()
    t["target_date"] = pd.to_datetime(t["target_date"], errors="coerce").dt.normalize()
    t["symbol"] = t["symbol"].astype(str).str.upper().str.strip()
    t["return_pct"] = _num(t.get("return_pct"))
    t = t[["target_date", "symbol", "return_pct"]].drop_duplicates(["target_date", "symbol"])

    e = evaluations.copy()
    e["target_date"] = pd.to_datetime(e["target_date"], errors="coerce").dt.normalize()
    e["symbol"] = e["symbol"].astype(str).str.upper().str.strip()
    e["close_abs_pct_error"] = _num(e.get("close_abs_pct_error"))
    e = e[["target_date", "symbol", "close_abs_pct_error"]].dropna(subset=["target_date", "symbol"])

    rows = []
    for _, r in p.sort_values(["target_date", "symbol"]).iterrows():
        target = r["target_date"]
        sym = r["symbol"]
        prior_t = t[(t["symbol"] == sym) & (t["target_date"] < target)].tail(20)
        prior_e = e[(e["symbol"] == sym) & (e["target_date"] < target)].tail(20)
        expected = (r["predicted_close"] / r["base_close"] - 1.0) * 100.0 if r.get("base_close", np.nan) else np.nan
        pred_range = ((r["predicted_high"] - r["predicted_low"]) / r["base_close"] * 100.0) if r.get("base_close", np.nan) else np.nan
        rows.append({
            "target_date": target, "symbol": sym,
            "rank": r.get("rank"), "score": r.get("score"),
            "confidence_score": r.get("confidence_score"),
            "prediction_spread": r.get("prediction_spread"),
            "expected_return_pct": expected,
            "predicted_range_pct": pred_range,
            "recent_symbol_return_pct": prior_t["return_pct"].mean() if not prior_t.empty else 0.0,
            "recent_symbol_win_rate": (prior_t["return_pct"] > 0).mean() if not prior_t.empty else 0.5,
            "recent_symbol_error_pct": prior_e["close_abs_pct_error"].mean() * 100.0 if not prior_e.empty else 0.0,
            "rank_percentile": 1.0 - (r.get("rank", 999) - 1.0) / 10.0,
        })
    x = pd.DataFrame(rows)
    if x.empty:
        return x
    x["rank_percentile"] = x["rank_percentile"].clip(0, 1)
    return x


def _fit(x: pd.DataFrame, y: pd.Series) -> dict:
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    extra = ExtraTreesClassifier(n_estimators=250, max_depth=10, min_samples_leaf=5, max_features=0.8, n_jobs=-1, random_state=42, class_weight="balanced")
    hgb.fit(x, y)
    extra.fit(x, y)
    return {"models": [hgb, extra], "weights": [0.7, 0.3], "features": FEATURES, "version": VERSION}


def _predict(bundle, x):
    weights = np.asarray(bundle["weights"], dtype=float)
    weights /= weights.sum()
    return sum(w * m.predict_proba(x)[:, 1] for w, m in zip(weights, bundle["models"]))


def train_challenger() -> dict:
    if not PREDICTIONS_FILE.exists() or not TRADES_FILE.exists():
        return {"status": "collecting", "reason": "stored predictions/trades unavailable"}
    pred = pd.read_csv(PREDICTIONS_FILE)
    trades = pd.read_csv(TRADES_FILE)
    evaluations = pd.read_csv(EVALUATIONS_FILE) if EVALUATIONS_FILE.exists() else pd.DataFrame()
    x = _base_features(pred, trades, evaluations)
    if x.empty:
        return {"status": "collecting", "trades": 0}
    actual = trades.copy()
    actual["target_date"] = pd.to_datetime(actual["target_date"], errors="coerce").dt.normalize()
    actual["symbol"] = actual["symbol"].astype(str).str.upper().str.strip()
    actual["return_pct"] = _num(actual["return_pct"])
    labels = actual[["target_date", "symbol", "return_pct"]].drop_duplicates(["target_date", "symbol"])
    ds = x.merge(labels, on=["target_date", "symbol"], how="inner").dropna(subset=FEATURES + ["return_pct"]).copy()
    ds["target"] = (ds["return_pct"] > 0).astype(int)
    sessions = sorted(ds["target_date"].dropna().unique())
    if len(ds) < MIN_TRADES or len(sessions) < MIN_SESSIONS or ds["target"].nunique() < 2:
        return {"status": "collecting", "trades": len(ds), "sessions": len(sessions)}

    split = sessions[max(1, int(len(sessions) * 0.70)) - 1]
    train = ds[ds["target_date"] <= split]
    val = ds[ds["target_date"] > split]
    if len(val) < MIN_VALIDATION_TRADES or train["target"].nunique() < 2:
        return {"status": "collecting", "trades": len(ds), "sessions": len(sessions)}

    bundle = _fit(train[FEATURES], train["target"])
    val = val.copy()
    val["quality_probability"] = _predict(bundle, val[FEATURES])
    val["baseline_probability"] = 0.5
    auc = float(roc_auc_score(val["target"], val["quality_probability"])) if val["target"].nunique() == 2 else np.nan
    top_n = max(1, int(np.ceil(len(val) * 0.5)))
    top = val.nlargest(top_n, "quality_probability")
    baseline = val.nlargest(top_n, "rank_percentile")
    top_win = float(top["target"].mean() * 100)
    base_win = float(baseline["target"].mean() * 100)
    lift = top_win - base_win
    ready = bool(len(ds) >= MIN_TRADES and len(sessions) >= MIN_SESSIONS and lift >= 2.0 and (pd.isna(auc) or auc >= 0.55))
    status = "promote" if ready else "validated_hold"
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    out = {"sessions": len(sessions), "trades": len(ds), "validation_trades": len(val),
           "auc": auc, "top_half_win_rate_pct": top_win, "baseline_top_half_win_rate_pct": base_win,
           "win_rate_lift_pct": lift, "production_ready": ready, "status": status}
    pd.DataFrame([out]).to_csv(SUMMARY_FILE, index=False)
    val.to_csv(VALIDATION_FILE, index=False)
    print(f"Trade quality challenger: trades={len(ds)}, sessions={len(sessions)}, AUC={auc:.3f}, win_lift={lift:.2f}pp, status={status}")
    return out


def latest_trade_quality_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_FILE.exists() or not SUMMARY_FILE.exists():
        return pd.DataFrame()
    summary = pd.read_csv(SUMMARY_FILE)
    if summary.empty or str(summary.iloc[-1].get("status", "")) != "promote":
        return pd.DataFrame()
    bundle = joblib.load(MODEL_FILE)
    pred = predictions.copy()
    if pred.empty:
        return pd.DataFrame()
    # Rebuild only the current candidates. Historical trades/evaluations are used
    # strictly before each target date, preventing future information leakage.
    trades = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
    evaluations = pd.read_csv(EVALUATIONS_FILE) if EVALUATIONS_FILE.exists() else pd.DataFrame()
    x = _base_features(pred, trades, evaluations)
    if x.empty:
        return pd.DataFrame()
    x["trade_quality_probability"] = _predict(bundle, x[FEATURES])
    return x[["symbol", "target_date", "trade_quality_probability", "recent_symbol_return_pct", "recent_symbol_win_rate", "recent_symbol_error_pct"]]


if __name__ == "__main__":
    print(train_challenger())
