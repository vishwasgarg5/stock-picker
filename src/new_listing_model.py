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
    x["target_close_return"] = g["close"].shift(-1) / x["close"] - 1.0
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
    x = x.dropna(subset=IPO_FEATURES + ["target_close_return"])

    status["symbols"] = int(x["symbol"].nunique())
    status["rows"] = int(len(x))
    if status["symbols"] < MIN_SYMBOLS or status["rows"] < MIN_ROWS:
        pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
        print(
            f"New-listing challenger collecting: {status['symbols']} symbols, "
            f"{status['rows']} rows; need {MIN_SYMBOLS} symbols and {MIN_ROWS} rows"
        )
        return status

    # Challenger only: never promoted automatically. The model is trained on
    # new-listing observations and must later pass out-of-sample validation.
    h = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=200, learning_rate=0.05,
        max_leaf_nodes=15, l2_regularization=1.0, random_state=42
    )
    e = ExtraTreesRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=4,
        max_features=0.8, n_jobs=-1, random_state=42
    )
    X, y = x[IPO_FEATURES], x["target_close_return"]
    h.fit(X, y)
    e.fit(X, y)
    joblib.dump({"models": [h, e], "weights": [0.7, 0.3], "version": "new_listing_challenger_v1",
                 "features": IPO_FEATURES}, MODEL_FILE)
    status["status"] = "trained_challenger"
    pd.DataFrame([status]).to_csv(STATUS_FILE, index=False)
    print(f"New-listing challenger trained: {status['symbols']} symbols, {status['rows']} rows")
    return status


if __name__ == "__main__":
    train_new_listing_challenger()
