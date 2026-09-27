from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ranking_diagnostics import run_ranking_diagnostics


if __name__ == "__main__":
    run_ranking_diagnostics()
