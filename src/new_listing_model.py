from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
NEW_LISTINGS_FILE = DATA / "new_listings.csv"
HISTORY_FILE = DATA / "ohlcv.csv"
MODEL_TARGETS = ["open", "high", "low", "close"]
MODEL_FILE = MODELS / "new_listing_close.joblib"
STATUS_FILE = DATA / "new_listing_model_status.csv"

IPO_FEATURES = [
    "age_sessions",
    "return_1d", "return_3d", "return_5d",
    "range_pct", "volatility5", "volume_ratio",
    "gap_pct", "close_position",
]
MIN_SYMBOLS = 3
MIN_ROWS = 100
MIN_AGE = 20


def _features(hist: pd.DataFrame, listing_dates: pd.Series) -> pd.DataFrame:
    x = hist.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x = x.dropna(subset=["date", "close", "open", "high", "low"])
    x = x.sort_values(["symbol", "date"]).copy()
    x["listing_date"] = x["symbol"].map(listing_dates)
    x["age_sessions"] = x.groupby("symbol").cumcount() + 1
    g = x.groupby("symbol", group_keys=False)
    x["return_1d"] = g["close"].pct_change()
    x["return_3d"] = g["close"].pct_change(3)
    x["return_5d"] = g["close"].pct_change(5)
    x["range_pct"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["volatility5"] = g["return_1d"].transform(lambda s: s.rolling(5, min_periods=3).std())
    vol5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    x["volume_ratio"] = x["volume"] / vol5.replace(0, np.nan)
    prev = g["close"].shift(1)
    x["gap_pct"] = (x["open"] / prev.replace(0, np.nan)) - 1.0
    x["close_position"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
    # One-step-ahead return; the current row contains information known at session close.
    for field in MODEL_TARGETS:
        x[f"target_{field}_return"] = g[field].shift(-1) / x["close"] - 1.0
    return x


def train_new_listing_challenger() -> dict:
    status = {
        "status": "collecting",
        "symbols": 0,
        "rows": 0,
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
    }
    if not NEW_LISTINGS_FILE.exists() or not HISTORY_FILE.exists():
        pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
        return status

    listings = pd.read_csv(NEW_LISTINGS_FILE)
    if listings.empty or not {"symbol", "listing_date"}.issubset(listings.columns):
        pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
        return status

    listings["symbol"] = listings["symbol"].astype(str).str.upper().str.strip()
    listings["listing_date"] = pd.to_datetime(listings["listing_date"], errors="coerce").dt.normalize()
    listings = listings.dropna(subset=["symbol", "listing_date"]).drop_duplicates("symbol")
    listing_dates = listings.set_index("symbol")["listing_date"]

    hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    x = _features(hist, listing_dates)
    x = x[x["symbol"].isin(listing_dates.index)]
    x = x[x["age_sessions"] >= MIN_AGE]
    x = x.dropna(subset=IPO_FEATURES + [f"target_{t}_return" for t in MODEL_TARGETS])

    status["symbols"] = int(x["symbol"].nunique())
    status["rows"] = int(len(x))
    if status["symbols"] < MIN_SYMBOLS or status["rows"] < MIN_ROWS:
        pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
        print(
            f"New-listing challenger collecting: {status['symbols']} symbols, "
            f"{status['rows']} rows; need {MIN_SYMBOLS} symbols and {MIN_ROWS} rows"
        )
        return status

    # Time-ordered holdout: the challenger must beat a zero-return baseline
    # before it can even be considered for production use.
    x = x.sort_values(["date", "symbol"]).reset_index(drop=True)
    cutoff = x["date"].quantile(0.80)
    train_x = x[x["date"] < cutoff]
    test_x = x[x["date"] >= cutoff]
    if len(train_x) < 60 or test_x.empty:
        pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
        return status

    def fit_models(frame):
        h = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=200, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=1.0, random_state=42
        )
        e = ExtraTreesRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=4,
            max_features=0.8, n_jobs=-1, random_state=42
        )
        h.fit(frame[IPO_FEATURES], frame["target_close_return"])
        e.fit(frame[IPO_FEATURES], frame["target_close_return"])
        return h, e

    h, e = fit_models(train_x)
    pred_h = h.predict(test_x[IPO_FEATURES])
    pred_e = e.predict(test_x[IPO_FEATURES])
    pred = 0.7 * pred_h + 0.3 * pred_e
    actual = test_x["target_close_return"].to_numpy()
    model_mae = float(np.mean(np.abs(pred - actual)))
    baseline_mae = float(np.mean(np.abs(actual)))
    status["holdout_rows"] = int(len(test_x))
    status["model_mae"] = model_mae
    status["baseline_mae"] = baseline_mae
    status["relative_improvement_pct"] = (
        (baseline_mae - model_mae) / baseline_mae * 100.0
        if baseline_mae > 0 else 0.0
    )

    # Refit on all available observations, but keep it as a challenger only.
    h, e = fit_models(x)
    joblib.dump({"models": [h, e], "weights": [0.7, 0.3], "version": "new_listing_challenger_v1",
                 "features": IPO_FEATURES, "holdout_model_mae": model_mae,
                 "holdout_baseline_mae": baseline_mae}, MODEL_FILE)
    status["status"] = "validated_challenger" if (
        baseline_mae > 0 and model_mae < baseline_mae * 0.99
    ) else "trained_challenger"

    pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
    print(f"New-listing challenger trained: {status['symbols']} symbols, {status['rows']} rows")
    return status

def challenger_ready() -> bool:
    if not STATUS_FILE.exists() or not all((MODELS / f"new_listing_{t}.joblib").exists() for t in MODEL_TARGETS):
        return False
    try:
        s = pd.read_csv(STATUS_FILE)
        return not s.empty and str(s.iloc[-1].get("status", "")) == "validated_challenger"
    except Exception:
        return False


def latest_challenger_features(hist: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not NEW_LISTINGS_FILE.exists():
        return pd.DataFrame()
    listings = pd.read_csv(NEW_LISTINGS_FILE)
    if listings.empty or not {"symbol", "listing_date", "calendar_age"}.issubset(listings.columns):
        return pd.DataFrame()
    listings["symbol"] = listings["symbol"].astype(str).str.upper().str.strip()
    listings["listing_date"] = pd.to_datetime(listings["listing_date"], errors="coerce").dt.normalize()
    listings["calendar_age"] = pd.to_numeric(listings["calendar_age"], errors="coerce")
    eligible = listings[
        listings["symbol"].isin([str(s).upper().strip() for s in symbols])
        & listings["calendar_age"].between(20, 59, inclusive="both")
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    dates = eligible.drop_duplicates("symbol").set_index("symbol")["listing_date"]
    x = _features(hist, dates)
    latest = x.sort_values("date").groupby("symbol", as_index=False).tail(1)
    latest = latest[latest["symbol"].isin(eligible["symbol"])]
    return latest.dropna(subset=IPO_FEATURES).copy()


def predict_challenger_ohlc(hist: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not challenger_ready():
        return pd.DataFrame()
    latest = latest_challenger_features(hist, symbols)
    if latest.empty:
        return pd.DataFrame()
    out = latest[["symbol", "date", "close", "age_sessions"]].copy()
    for target in MODEL_TARGETS:
        bundle = joblib.load(MODELS / f"new_listing_{target}.joblib")
        models = bundle["models"]
        weights = np.asarray(bundle.get("weights", [0.7, 0.3]), dtype=float)
        weights = weights / weights.sum()
        pred = sum(w * m.predict(latest[IPO_FEATURES]) for w, m in zip(weights, models))
        out[f"predicted_{target}"] = latest["close"] * (1.0 + pred)
    out["predicted_high"] = out[["predicted_high", "predicted_open", "predicted_close"]].max(axis=1)
    out["predicted_low"] = out[["predicted_low", "predicted_open", "predicted_close"]].min(axis=1)
    return out

if __name__ == "__main__":
    train_new_listing_challenger()
