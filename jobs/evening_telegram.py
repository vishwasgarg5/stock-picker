from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.telegram import send_evening, send_paper_trading


if __name__ == "__main__":
    send_evening()
    send_paper_trading()
