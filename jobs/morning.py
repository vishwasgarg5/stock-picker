from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.universe import update_universe
from src.pipeline import run_morning


if __name__ == "__main__":
    update_universe()
    run_morning()
