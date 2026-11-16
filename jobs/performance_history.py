from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.performance_history import update_performance_history

if __name__ == "__main__":
    update_performance_history()
