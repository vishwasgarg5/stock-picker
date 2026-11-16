from pathlib import Path
import sys

import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.universe import update_universe
from src.pipeline import run_morning


def stored_prediction_for_next_session() -> bool:
    """True only when a complete prediction exists for the next NSE session."""
    predictions_path = ROOT / "data" / "predictions.csv"
    history_path = ROOT / "data" / "ohlcv.csv"
    if not predictions_path.exists() or not history_path.exists():
        return False
    try:
        predictions = pd.read_csv(predictions_path)
        history = pd.read_csv(history_path, usecols=["date"])
        prediction_dates = pd.to_datetime(predictions.get("target_date"), errors="coerce").dropna()
        history_dates = pd.to_datetime(history["date"], errors="coerce").dropna()
        if prediction_dates.empty or history_dates.empty:
            return False

        last_market_date = history_dates.dt.normalize().max()
        calendar = mcal.get_calendar("XNSE")
        start = last_market_date + pd.Timedelta(days=1)
        schedule = calendar.schedule(start_date=start.date(), end_date=(start + pd.Timedelta(days=14)).date())
        if schedule.empty:
            return False
        next_session = pd.Timestamp(schedule.index[0]).normalize()

        rows = predictions[
            pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize() == next_session
        ]
        return len(rows) == 10 and rows["symbol"].astype(str).nunique() == 10
    except Exception as exc:
        print(f"Stored prediction check failed; running normal morning pipeline: {exc}")
        return False


if __name__ == "__main__":
    if stored_prediction_for_next_session():
        print("Complete prediction for the next NSE session already exists; keeping it unchanged.")
    else:
        update_universe()
        run_morning()
