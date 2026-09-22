from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.universe import update_universe
from src.pipeline import run_morning


def stored_prediction_exists() -> bool:
    """Return True when a complete stored prediction session already exists."""
    import pandas as pd

    path = ROOT / "data" / "predictions.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        if "target_date" not in df.columns:
            return False
        dates = pd.to_datetime(df["target_date"], errors="coerce").dropna()
        if dates.empty:
            return False
        latest = dates.dt.normalize().max()
        return len(df[df["target_date"].notna() & (pd.to_datetime(df["target_date"], errors="coerce").dt.normalize() == latest)]) == 10
    except Exception:
        return False


if __name__ == "__main__":
    # A manual/scheduled rerun with an already stored prediction must not
    # recalculate or overwrite that prediction.
    if stored_prediction_exists():
        print("Complete stored prediction session found; keeping prediction unchanged.")
    else:
        update_universe()
        run_morning()
