from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from .pipeline import FEATURE_COLUMNS, features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_FILE = ROOT / "models" / "directional_challenger.joblib"
VALIDATION_FILE = DATA / "directional_model_validation.csv"
SUMMARY_FILE = DATA / "directional_model_validation_summary.csv"
HISTORY_FILE = DATA / "ohlcv.csv"

MIN_ROWS = 500
MIN_SESSIONS = 12
RECENT_WINDOWS = (5, 10, 20)


def _dataset(hist: pd.DataFrame) -> pd.DataFrame:
    x = features(hist).sort_values(["symbol", "date"]).copy()
    g = x.groupby("symbol", group_keys=False)
    next_return = g["close"].shift(-1) / x["close"] - 1.0
    x["target_direction"] = (next_return > 0).astype(int)
    # Ignore essentially flat moves; they are not useful directional training
    # examples and otherwise make the class boundary artificially noisy.
    x["target_return"] = next_return
    x = x.dropna(subset=FEATURE_COLUMNS + ["target_return"]).copy()
    x = x[x["target_return"].abs() >= 0.001].copy()
    return x


def _fit(x: pd.DataFrame, y: pd.Series) -> dict:
    hgb = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42
    )
    extra = ExtraTreesClassifier(
        n_estimators=250, max_depth=14, min_samples_leaf=5,
        max_features=0.8, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    hgb.fit(x, y)
    extra.fit(x, y)
    return {
        "models": [hgb, extra],
        "weights": [0.65, 0.35],
        "version": "directional_challenger_v1",
        "features": FEATURE_COLUMNS,
    }


def _predict(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    weights = np.asarray(bundle.get("weights", [0.5, 0.5]), dtype=float)
    weights = weights / weights.sum()
    probs = np.column_stack([m.predict_proba(x)[:, 1] for m in bundle["models"]])
    return probs @ weights


def _recent(val: pd.DataFrame) -> dict:
    out = {}
    for n in RECENT_WINDOWS:
        w = val.tail(n)
        out[f"recent_{n}_sessions"] = len(w)
        out[f"recent_{n}_accuracy_pct"] = (
            float(w["model_accuracy_pct"].mean()) if len(w) == n else np.nan
        )
        out[f"recent_{n}_lift_pct"] = (
            float(w["accuracy_lift_pct"].mean()) if len(w) == n else np.nan
        )
    return out


def train_challenger(hist: pd.DataFrame) -> dict:
    ds = _dataset(hist)
    if len(ds) < MIN_ROWS:
        return {"status": "collecting", "rows": len(ds)}

    dates = sorted(ds["date"].unique())
    split_idx = max(1, int(len(dates) * 0.70)) - 1
    split = dates[split_idx]
    train = ds[ds["date"] <= split]
    test = ds[ds["date"] > split]
    if train.empty or test.empty:
        return {"status": "collecting", "rows": len(ds)}

    bundle = _fit(train[FEATURE_COLUMNS], train["target_direction"])
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)

    rows = []
    for date, g in test.groupby("date", sort=True):
        if len(g) < 10:
            continue
        p = _predict(bundle, g[FEATURE_COLUMNS])
        actual = g["target_direction"].to_numpy()
        model_acc = float(((p >= 0.5).astype(int) == actual).mean() * 100.0)
        baseline_acc = float(max(actual.mean(), 1.0 - actual.mean()) * 100.0)
        rows.append({
            "target_date": date,
            "rows": len(g),
            "model_accuracy_pct": model_acc,
            "baseline_accuracy_pct": baseline_acc,
            "accuracy_lift_pct": model_acc - baseline_acc,
        })

    val = pd.DataFrame(rows)
    if val.empty:
        return {"status": "collecting", "rows": len(ds), "sessions": 0}

    val.to_csv(VALIDATION_FILE, index=False)
    sessions = len(val)
    mean_acc = float(val["model_accuracy_pct"].mean())
    lift = float(val["accuracy_lift_pct"].mean())
    recent = _recent(val)
    ready = bool(
        sessions >= MIN_SESSIONS
        and mean_acc >= 52.0
        and lift > 0
        and all(
            recent[f"recent_{n}_sessions"] == n
            and pd.notna(recent[f"recent_{n}_lift_pct"])
            and recent[f"recent_{n}_lift_pct"] > 0
            for n in RECENT_WINDOWS
        )
    )
    summary = {
        "sessions": sessions,
        "rows": int(val["rows"].sum()),
        "mean_accuracy_pct": mean_acc,
        "mean_accuracy_lift_pct": lift,
        "production_ready": ready,
        "status": "promote" if ready else ("hold" if sessions >= MIN_SESSIONS else "collecting"),
        **recent,
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)
    print(
        f"Directional challenger: sessions={sessions}, accuracy={mean_acc:.2f}%, "
        f"lift={lift:.2f}pp, recent20={recent.get('recent_20_lift_pct', np.nan):.2f}pp, "
        f"status={summary['status']}"
    )
    return summary


def latest_direction_scores(hist: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_FILE.exists() or not SUMMARY_FILE.exists():
        return pd.DataFrame()
    s = pd.read_csv(SUMMARY_FILE)
    if s.empty:
        return pd.DataFrame()
    ready = str(s.iloc[-1].get("production_ready", False)).strip().lower() == "true"
    if not ready:
        return pd.DataFrame()
    bundle = joblib.load(MODEL_FILE)
    x = features(hist).sort_values("date").groupby("symbol", as_index=False).tail(1)
    x = x.dropna(subset=FEATURE_COLUMNS)
    if x.empty:
        return pd.DataFrame()
    p = _predict(bundle, x[FEATURE_COLUMNS])
    return pd.DataFrame({
        "symbol": x["symbol"].astype(str).str.upper().str.strip().values,
        "direction_probability": p,
        "direction_score_model": p * 100.0,
    })


if __name__ == "__main__":
    hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    train_challenger(hist)
