from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_evening
from src.paper_trading import run_paper_trading


if __name__ == "__main__":
    run_evening()
    run_paper_trading()
