from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from .pipeline import DATA, FEATURE_COLUMNS, features

MODEL_FILE = DATA.parent / "models" / "ranking_challenger.joblib"
VALIDATION_FILE = DATA / "ranking_model_validation.csv"
HISTORY_FILE = DATA / "ohlcv.csv"

RANK_FEATURES = FEATURE_COLUMNS + ["market_momentum", "market_volatility", "cross_sectional_momentum"]
MIN_ROWS = 500
MIN_SESSIONS = 12
MIN_ROWS_VALIDATION = 100


def _build_dataset(hist: pd.DataFrame) -> pd.DataFrame:
    x = features(hist).sort_values(["symbol", "date"]).copy()
    g = x.groupby("symbol", group_keys=False)
    x["target_return"] = g["close"].shift(-1) / x["close"] - 1.0
    daily = x.groupby("date")["close"].median().sort_index()
    market_ret = daily.pct_change()
    market_vol = market_ret.rolling(20).std()
    x["market_momentum"] = x["date"].map(daily.pct_change(20))
    x["market_volatility"] = x["date"].map(market_vol)
    x["cross_sectional_momentum"] = x.groupby("date")["return_20d"].transform(
        lambda s: s.rank(pct=True)
    )
    return x.dropna(subset=RANK_FEATURES + ["target_return"]).copy()


def _fit(x: pd.DataFrame, y: pd.Series) -> dict:
    hgb = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=250, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=42
    )
    extra = ExtraTreesRegressor(
        n_estimators=250, max_depth=14, min_samples_leaf=5,
        max_features=0.8, n_jobs=-1, random_state=42
    )
    hgb.fit(x, y)
    extra.fit(x, y)
    return {"models": [hgb, extra], "weights": [0.7, 0.3],
            "version": "ranking_challenger_v1", "features": RANK_FEATURES}


def _predict(bundle: dict, x: pd.DataFrame) -> np.ndarray:
    weights = np.asarray(bundle["weights"], dtype=float)
    weights /= weights.sum()
    return sum(w * m.predict(x) for w, m in zip(weights, bundle["models"]))


def train_challenger(hist: pd.DataFrame) -> dict:
    ds = _build_dataset(hist)
    if len(ds) < MIN_ROWS:
        print(f"Ranking challenger: collecting data ({len(ds)} rows)")
        return {"status": "collecting", "rows": len(ds)}

    # Train only on the earlier chronological portion. Validation below uses
    # strictly later sessions, preventing future-return leakage.
    dates = sorted(ds["date"].dropna().unique())
    split = dates[max(1, int(len(dates) * 0.70)) - 1]
    train = ds[ds["date"] <= split].copy()
    if len(train) < MIN_ROWS // 2:
        return {"status": "collecting", "rows": len(ds)}

    bundle = _fit(train[RANK_FEATURES], train["target_return"])
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    result = validate_challenger(ds[ds["date"] > split].copy(), bundle)
    return result


def validate_challenger(ds: pd.DataFrame, bundle: dict) -> dict:
    rows = []
    for target_date, g in ds.groupby("date", sort=True):
        if len(g) < 10:
            continue
        pred = _predict(bundle, g[RANK_FEATURES])
        y = g["target_return"].to_numpy()
        order = np.argsort(-pred)
        base_order = np.argsort(-g["return_20d"].to_numpy())
        n = min(10, len(g))
        top = y[order[:n]]
        base = y[base_order[:n]]
        rows.append({
            "target_date": target_date,
            "sessions": 1,
            "rows": len(g),
            "model_top10_return_pct": float(np.mean(top) * 100),
            "baseline_top10_return_pct": float(np.mean(base) * 100),
            "model_top10_positive_pct": float(np.mean(top > 0) * 100),
            "baseline_top10_positive_pct": float(np.mean(base > 0) * 100),
        })
    val = pd.DataFrame(rows)
    if val.empty:
        return {"status": "collecting", "sessions": 0, "rows": 0}

    val["return_lift_pct"] = val["model_top10_return_pct"] - val["baseline_top10_return_pct"]
    val.to_csv(VALIDATION_FILE, index=False)
    sessions = len(val)
    total_rows = int(val["rows"].sum())
    lift = float(val["return_lift_pct"].mean())
    positive_lift = float(
        val["model_top10_positive_pct"].mean() -
        val["baseline_top10_positive_pct"].mean()
    )
    promoted = (
        sessions >= MIN_SESSIONS and
        total_rows >= MIN_ROWS_VALIDATION and
        lift > 0 and positive_lift >= 0
    )
    status = "promote" if promoted else (
        "collecting" if sessions < MIN_SESSIONS or total_rows < MIN_ROWS_VALIDATION
        else "hold"
    )
    summary = pd.DataFrame([{
        "sessions": sessions, "rows": total_rows,
        "mean_top10_return_lift_pct": lift,
        "positive_rate_lift_pct": positive_lift,
        "promotion_evidence": promoted,
        "status": status,
    }])
    summary.to_csv(DATA / "ranking_model_validation_summary.csv", index=False)
    print(
        f"Ranking challenger: sessions={sessions}, rows={total_rows}, "
        f"return_lift={lift:.3f}%, status={status}"
    )
    return summary.iloc[0].to_dict()

def latest_rank_scores(hist: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_FILE.exists() or not VALIDATION_FILE.exists():
        return pd.DataFrame()
    summary_file = DATA / "ranking_model_validation_summary.csv"
    if not summary_file.exists():
        return pd.DataFrame()
    s = pd.read_csv(summary_file)
    if s.empty or str(s.iloc[-1].get("status", "")) != "promote":
        return pd.DataFrame()
    bundle = joblib.load(MODEL_FILE)
    x = _build_dataset(hist)
    latest_date = pd.to_datetime(hist["date"]).max().normalize()
    x = x[x["date"] == latest_date].copy()
    if x.empty:
        return pd.DataFrame()
    x["ranking_model_score"] = _predict(bundle, x[RANK_FEATURES])
    return x[["symbol", "ranking_model_score"]]


if __name__ == "__main__":
    hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    train_challenger(hist)
