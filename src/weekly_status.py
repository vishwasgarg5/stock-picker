from __future__ import annotations

from pathlib import Path
import html
import os

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PERFORMANCE = DATA / "performance_history.csv"
RANKING = DATA / "ranking_validation.csv"
RANKING_HORIZONS = DATA / "ranking_validation_horizons.csv"
PAPER_DAILY = DATA / "portfolio_daily.csv"
PAPER_METRICS = DATA / "trading_strategy_metrics.csv"
WEEKLY_SENT = DATA / "telegram_weekly_sent.csv"


def _fmt(v, suffix=""):
    try:
        return f"{float(v):.2f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _send(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured")
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    r.raise_for_status()
    if not r.json().get("ok"):
        raise RuntimeError(f"Telegram API error: {r.json()}")


def _already_sent(week_start: pd.Timestamp) -> bool:
    if not WEEKLY_SENT.exists():
        return False
    x = pd.read_csv(WEEKLY_SENT)
    return "week_start" in x and x["week_start"].astype(str).eq(week_start.date().isoformat()).any()


def _record_sent(week_start: pd.Timestamp) -> None:
    old = pd.read_csv(WEEKLY_SENT) if WEEKLY_SENT.exists() else pd.DataFrame(columns=["week_start"])
    row = pd.DataFrame({"week_start": [week_start.date().isoformat()]})
    pd.concat([old, row], ignore_index=True).drop_duplicates("week_start").to_csv(WEEKLY_SENT, index=False)


def build_weekly_message() -> str:
    if not PERFORMANCE.exists():
        raise RuntimeError("performance_history.csv does not exist yet")

    perf = pd.read_csv(PERFORMANCE)
    if perf.empty or "target_date" not in perf:
        raise RuntimeError("No performance history available")

    perf["target_date"] = pd.to_datetime(perf["target_date"], errors="coerce").dt.normalize()
    perf = perf.dropna(subset=["target_date"]).sort_values("target_date")
    latest = perf["target_date"].max()
    week_start = latest - pd.Timedelta(days=latest.weekday())
    week = perf[perf["target_date"].between(week_start, latest)].copy()

    latest_row = week.iloc[-1] if not week.empty else perf.iloc[-1]
    sessions = len(week)

    lines = [
        "<b>STOCK PICKER · WEEKLY MODEL STATUS</b>",
        f"{week_start:%d-%b-%Y} → {latest:%d-%b-%Y}",
        "",
        "<b>MODEL STATUS</b>",
        f"Evaluated sessions   {sessions}",
        f"Total sessions       {int(latest_row.get('sessions', len(perf)))}",
        f"Latest Close MAPE    {_fmt(latest_row.get('close_mape_pct'), '%')}",
        f"Baseline Close MAPE  {_fmt(latest_row.get('baseline_close_mape_pct'), '%')}",
        f"Close improvement    {_fmt(latest_row.get('close_improvement_vs_baseline_pct'), '%')}",
        f"Direction accuracy   {_fmt(latest_row.get('direction_accuracy_pct'), '%')}",
        "",
        "<b>DAY-WISE PROGRESS</b>",
        "<pre>",
        "Date       | Sessions | Close MAPE | Baseline | Direction",
        "---------------------------------------------------------",
    ]

    for _, r in week.iterrows():
        lines.append(
            f"{r['target_date']:%d-%b}    | "
            f"{int(r.get('predictions_evaluated', 0)):>8} | "
            f"{_fmt(r.get('close_mape_pct'), '%'):>10} | "
            f"{_fmt(r.get('baseline_close_mape_pct'), '%'):>8} | "
            f"{_fmt(r.get('direction_accuracy_pct'), '%'):>9}"
        )
    lines.append("</pre>")

    if RANKING_HORIZONS.exists():
        rank = pd.read_csv(RANKING_HORIZONS)
        if not rank.empty and "prediction_date" in rank:
            rank["prediction_date"] = pd.to_datetime(rank["prediction_date"], errors="coerce").dt.normalize()
            rw = rank[rank["prediction_date"].between(week_start, latest)]
            lines += ["", "<b>RANKING PROGRESS</b>"]
            for horizon in (1, 5, 10, 20):
                h = rw[rw["horizon_sessions"].eq(horizon)]
                top10 = h[h["group"].eq("TOP10")]
                if top10.empty:
                    continue
                r = top10.iloc[-1]
                gap = r.get("lift_vs_comparator_pct")
                universe_lift = r.get("lift_vs_universe_pct")
                lines.append(
                    "Top-10 {:>2}D | vs 11-20 {} | vs universe {}".format(
                        horizon, _fmt(gap, "%"), _fmt(universe_lift, "%")
                    )
                )


    mfe_path = DATA / "ranking_validation_mfe_mae.csv"
    if mfe_path.exists():
        mfe = pd.read_csv(mfe_path)
        if not mfe.empty and "prediction_date" in mfe:
            mfe["prediction_date"] = pd.to_datetime(mfe["prediction_date"], errors="coerce").dt.normalize()
            mw = mfe[mfe["prediction_date"].between(week_start, latest)]
            top10 = mw[mw["group"].eq("TOP10")]
            if not top10.empty:
                lines += ["", "<b>MFE / MAE</b>"]
                for horizon in (5, 10):
                    mh = top10[top10["horizon_sessions"].eq(horizon)]
                    if mh.empty:
                        continue
                    r = mh.iloc[-1]
                    lines.append(
                        "Top-10 {:>2}D | Return {} | MFE {} | MAE {}".format(
                            horizon,
                            _fmt(r.get("mean_final_return_pct"), "%"),
                            _fmt(r.get("mean_mfe_pct"), "%"),
                            _fmt(r.get("mean_mae_pct"), "%"),
                        )
                    )

    stability_path = DATA / "ranking_validation_stability.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        if not stability.empty and "prediction_date" in stability:
            stability["prediction_date"] = pd.to_datetime(stability["prediction_date"], errors="coerce").dt.normalize()
            sw = stability[stability["prediction_date"].between(week_start, latest)]
            if not sw.empty:
                r = sw.iloc[-1]
                lines += [
                    "",
                    "<b>RANKING STABILITY</b>",
                    "Latest Top-5 overlap   {}".format(_fmt(r.get("top5_overlap_pct"), "%")),
                    "Latest Top-10 overlap  {}".format(_fmt(r.get("top10_overlap_pct"), "%")),
                    "Mean rank movement      {}".format(_fmt(r.get("mean_abs_rank_change"))),
                ]

    turnover_path = DATA / "ranking_validation_turnover.csv"
    if turnover_path.exists():
        turnover = pd.read_csv(turnover_path)
        if not turnover.empty and "prediction_date" in turnover:
            turnover["prediction_date"] = pd.to_datetime(turnover["prediction_date"], errors="coerce").dt.normalize()
            tw = turnover[turnover["prediction_date"].between(week_start, latest)]
            if not tw.empty:
                r = tw.iloc[-1]
                lines += [
                    "",
                    "<b>RANKING TURNOVER</b>",
                    "Latest Top-5 turnover   {}".format(_fmt(r.get("top5_turnover_pct"), "%")),
                    "Latest Top-10 turnover  {}".format(_fmt(r.get("top10_turnover_pct"), "%")),
                    "Latest Top-20 turnover  {}".format(_fmt(r.get("top20_turnover_pct"), "%")),
                ]

    if PAPER_DAILY.exists():
        daily = pd.read_csv(PAPER_DAILY)
        if not daily.empty and "target_date" in daily:
            daily["target_date"] = pd.to_datetime(daily["target_date"], errors="coerce").dt.normalize()
            dw = daily[daily["target_date"].between(week_start, latest)]
            if not dw.empty:
                pnl = pd.to_numeric(dw.get("net_pnl"), errors="coerce").sum()
                ret = pd.to_numeric(dw.get("daily_return_pct"), errors="coerce").sum()
                lines += [
                    "",
                    "<b>PAPER TRADING PROGRESS</b>",
                    f"Week net P/L             ₹{_fmt(pnl)}",
                    f"Sum of daily returns      {_fmt(ret, '%')}",
                    f"Latest portfolio value    ₹{_fmt(dw.iloc[-1].get('portfolio_value'))}",
                    f"Latest drawdown            {_fmt(dw.iloc[-1].get('drawdown_pct'), '%')}",
                ]

    if PAPER_METRICS.exists():
        pm = pd.read_csv(PAPER_METRICS)
        if not pm.empty:
            r = pm.iloc[-1]
            lines += [
                "",
                "<b>LEARNING STATUS</b>",
                f"Learning rows             {r.get('learning_rows', '-')}",
                f"Learned model active      {r.get('learned_model_active', '-')}",
            ]

    lines += [
        "",
        "<b>NEXT VALIDATION FOCUS</b>",
        "• Continue collecting genuine out-of-sample sessions.",
        "• Monitor Close vs previous-close baseline.",
        "• Monitor Top-10 vs ranks 11-20.",
        "• Do not promote confidence/learning changes without validation evidence.",
    ]
    return "\n".join(lines)


def send_weekly_status() -> None:
    if not PERFORMANCE.exists():
        print("No performance history; weekly status skipped.")
        return
    perf = pd.read_csv(PERFORMANCE)
    if perf.empty:
        print("No performance sessions; weekly status skipped.")
        return
    perf["target_date"] = pd.to_datetime(perf["target_date"], errors="coerce").dt.normalize()
    latest = perf["target_date"].dropna().max()
    if pd.isna(latest):
        print("No valid performance date; weekly status skipped.")
        return
    week_start = latest - pd.Timedelta(days=latest.weekday())
    if os.environ.get("FORCE_TELEGRAM", "").lower() not in {"1", "true", "yes"} and _already_sent(week_start):
        print(f"Weekly Telegram already sent for week starting {week_start.date()}; skipping.")
        return
    message = build_weekly_message()
    _send(message)
    _record_sent(week_start)
    print(f"Weekly model status sent for week starting {week_start.date()}")


if __name__ == "__main__":
    send_weekly_status()
