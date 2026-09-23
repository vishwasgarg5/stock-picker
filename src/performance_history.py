from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVALUATIONS_FILE = DATA / "evaluations.csv"
PAPER_METRICS_FILE = DATA / "trading_strategy_metrics.csv"
PAPER_DAILY_FILE = DATA / "portfolio_daily.csv"
PERFORMANCE_FILE = DATA / "performance_history.csv"


def _mean_pct(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean() * 100.0) if not values.empty else float("nan")


def _session_metrics(evals: pd.DataFrame) -> pd.DataFrame:
    required = {"target_date", "open_abs_pct_error", "high_abs_pct_error",
                "low_abs_pct_error", "close_abs_pct_error"}
    missing = required - set(evals.columns)
    if missing:
        raise RuntimeError(f"Performance history: evaluations missing columns {sorted(missing)}")

    x = evals.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["target_date"]).copy()

    rows = []
    for target_date, group in x.groupby("target_date", sort=True):
        model = {
            "open_mape_pct": _mean_pct(group, "open_abs_pct_error"),
            "high_mape_pct": _mean_pct(group, "high_abs_pct_error"),
            "low_mape_pct": _mean_pct(group, "low_abs_pct_error"),
            "close_mape_pct": _mean_pct(group, "close_abs_pct_error"),
            "baseline_open_mape_pct": _mean_pct(group, "baseline_open_abs_pct_error"),
            "baseline_high_mape_pct": _mean_pct(group, "baseline_high_abs_pct_error"),
            "baseline_low_mape_pct": _mean_pct(group, "baseline_low_abs_pct_error"),
            "baseline_close_mape_pct": _mean_pct(group, "baseline_close_abs_pct_error"),
        }
        model["overall_mape_pct"] = float(np.nanmean([
            model["open_mape_pct"], model["high_mape_pct"],
            model["low_mape_pct"], model["close_mape_pct"]
        ]))
        baseline = model["baseline_close_mape_pct"]
        model["close_improvement_vs_baseline_pct"] = (
            (baseline - model["close_mape_pct"]) / baseline * 100.0
            if pd.notna(baseline) and baseline > 0 and pd.notna(model["close_mape_pct"])
            else float("nan")
        )
        direction = pd.to_numeric(group.get("close_direction_correct"), errors="coerce")
        model["direction_accuracy_pct"] = float(direction.mean() * 100.0) if direction is not None and direction.notna().any() else float("nan")
        model["predictions_evaluated"] = int(len(group))
        model["stocks_evaluated"] = int(group["symbol"].nunique()) if "symbol" in group else int(len(group))
        rows.append({"target_date": target_date, **model})

    history = pd.DataFrame(rows).sort_values("target_date").reset_index(drop=True)
    history["sessions"] = np.arange(1, len(history) + 1)

    # Session-based rolling metrics avoid calendar-day distortion around weekends/holidays.
    for window in (5, 10, 20):
        history[f"close_mape_{window}s_pct"] = history["close_mape_pct"].rolling(window, min_periods=1).mean()
        history[f"baseline_close_mape_{window}s_pct"] = history["baseline_close_mape_pct"].rolling(window, min_periods=1).mean()
        history[f"close_improvement_{window}s_pct"] = np.where(
            history[f"baseline_close_mape_{window}s_pct"] > 0,
            (history[f"baseline_close_mape_{window}s_pct"] - history[f"close_mape_{window}s_pct"]) /
            history[f"baseline_close_mape_{window}s_pct"] * 100.0,
            np.nan,
        )
        history[f"direction_accuracy_{window}s_pct"] = history["direction_accuracy_pct"].rolling(window, min_periods=1).mean()

    return history


def _attach_paper_metrics(history: pd.DataFrame) -> pd.DataFrame:
    result = history.copy()

    # Daily paper-trading series provides session-level P/L, cumulative return,
    # drawdown and win rate for historical sessions.
    if PAPER_DAILY_FILE.exists():
        daily = pd.read_csv(PAPER_DAILY_FILE)
        if not daily.empty and "target_date" in daily:
            daily["target_date"] = pd.to_datetime(daily["target_date"], errors="coerce").dt.normalize()
            daily_cols = ["target_date", "net_pnl", "daily_return_pct", "portfolio_value",
                          "cumulative_return_pct", "drawdown_pct", "win_rate_pct", "trades"]
            daily = daily[[c for c in daily_cols if c in daily.columns]]
            result = result.merge(daily, on="target_date", how="left")

    # Cumulative strategy/learning status from the latest metrics file.
    if PAPER_METRICS_FILE.exists():
        paper = pd.read_csv(PAPER_METRICS_FILE)
        if not paper.empty and "as_of" in paper:
            cols = ["as_of", "sessions", "total_return_pct", "max_drawdown_pct",
                    "avg_trade_return_pct", "profit_factor", "learning_rows",
                    "learned_model_active"]
            paper = paper[[c for c in cols if c in paper.columns]].rename(columns={"as_of": "target_date"})
            # Normalize the summary date before merging with the datetime session key.
            paper["target_date"] = pd.to_datetime(paper["target_date"], errors="coerce").dt.normalize()
            result = result.merge(paper, on="target_date", how="left", suffixes=("", "_summary"))

    return result

def update_performance_history() -> pd.DataFrame:
    if not EVALUATIONS_FILE.exists():
        print("No evaluations file; performance history not updated.")
        return pd.DataFrame()

    evals = pd.read_csv(EVALUATIONS_FILE)
    if evals.empty:
        print("No evaluations yet; performance history not updated.")
        return pd.DataFrame()

    history = _session_metrics(evals)
    history = _attach_paper_metrics(history)

    # Keep one immutable row per evaluated market session.
    if PERFORMANCE_FILE.exists():
        old = pd.read_csv(PERFORMANCE_FILE)
        if not old.empty and "target_date" in old:
            old["target_date"] = pd.to_datetime(old["target_date"], errors="coerce").dt.normalize()
            combined = pd.concat([old, history], ignore_index=True, sort=False)
            combined = combined.sort_values("target_date").drop_duplicates("target_date", keep="last")
            history = combined.reset_index(drop=True)
            history["sessions"] = np.arange(1, len(history) + 1)

            for window in (5, 10, 20):
                history[f"close_mape_{window}s_pct"] = history["close_mape_pct"].rolling(window, min_periods=1).mean()
                history[f"baseline_close_mape_{window}s_pct"] = history["baseline_close_mape_pct"].rolling(window, min_periods=1).mean()
                history[f"close_improvement_{window}s_pct"] = np.where(
                    history[f"baseline_close_mape_{window}s_pct"] > 0,
                    (history[f"baseline_close_mape_{window}s_pct"] - history[f"close_mape_{window}s_pct"]) /
                    history[f"baseline_close_mape_{window}s_pct"] * 100.0,
                    np.nan,
                )
                history[f"direction_accuracy_{window}s_pct"] = history["direction_accuracy_pct"].rolling(window, min_periods=1).mean()

    history.to_csv(PERFORMANCE_FILE, index=False)

    latest = history.iloc[-1]
    print(
        "Performance history updated: "
        f"sessions={int(latest['sessions'])}, date={pd.Timestamp(latest['target_date']).date()}, "
        f"close_mape={latest['close_mape_pct']:.3f}%, "
        f"baseline={latest['baseline_close_mape_pct']:.3f}%, "
        f"improvement_vs_baseline={latest['close_improvement_vs_baseline_pct']:.2f}%, "
        f"direction={latest['direction_accuracy_pct']:.2f}%"
    )
    if "close_mape_5s_pct" in latest:
        print(
            f"Rolling 5 sessions: close_mape={latest['close_mape_5s_pct']:.3f}%, "
            f"baseline={latest['baseline_close_mape_5s_pct']:.3f}%, "
            f"improvement={latest['close_improvement_5s_pct']:.2f}%"
        )
    return history


if __name__ == "__main__":
    update_performance_history()
