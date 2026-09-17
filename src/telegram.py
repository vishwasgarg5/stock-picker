from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"
PAPER_TRADES_FILE = DATA / "paper_trades.csv"
PORTFOLIO_FILE = DATA / "portfolio_daily.csv"
UNIVERSE_FILE = DATA / "universe.csv"
SENT_FILE = DATA / "telegram_sent.csv"
EVENING_SENT_FILE = DATA / "telegram_evening_sent.csv"
PAPER_SENT_FILE = DATA / "telegram_paper_sent.csv"


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _message_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _send(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


def _already_sent(path: Path, target_date: pd.Timestamp, message: str) -> bool:
    if not path.exists():
        return False
    sent = pd.read_csv(path)
    if "target_date" not in sent.columns or "message_hash" not in sent.columns:
        return False
    target = target_date.normalize()
    dates = pd.to_datetime(sent["target_date"], errors="coerce").dt.normalize()
    return ((dates == target) & sent["message_hash"].astype(str).eq(_message_hash(message))).any()


def _record_sent(path: Path, target_date: pd.Timestamp, message: str) -> None:
    columns = ["target_date", "message_hash"]
    if path.exists():
        sent = pd.read_csv(path)
        if not set(columns).issubset(sent.columns):
            sent = pd.DataFrame(columns=columns)
    else:
        sent = pd.DataFrame(columns=columns)
    new_row = pd.DataFrame({"target_date": [target_date.date().isoformat()], "message_hash": [_message_hash(message)]})
    sent = pd.concat([sent, new_row], ignore_index=True).drop_duplicates(columns, keep="last")
    sent.to_csv(path, index=False)


def _universe_count() -> int:
    if not UNIVERSE_FILE.exists():
        raise RuntimeError("Universe file does not exist")
    universe = pd.read_csv(UNIVERSE_FILE)
    if "symbol" not in universe.columns:
        raise RuntimeError("Universe file has no symbol column")
    symbols = universe["symbol"].astype(str).str.strip()
    count = symbols[symbols.ne("") & symbols.ne("nan")].nunique()
    if count <= 0:
        raise RuntimeError("Universe contains no stocks")
    return int(count)


def build_morning_message(predictions: pd.DataFrame, target_date: pd.Timestamp) -> str:
    rows = predictions[predictions["target_date"].dt.normalize() == target_date.normalize()].sort_values("rank").head(10)
    if len(rows) < 10:
        raise RuntimeError(f"Expected 10 predictions for {target_date.date()}, found {len(rows)}")
    total_stocks = _universe_count()
    lines = [
        "<b>STOCK PICKER</b>",
        f"{target_date:%d-%b-%Y} | TOP 10 / {total_stocks}",
        "",
        "<pre>",
        "Index   | Open      | High      | Low       | Close",
        "-----------------------------------------------------",
    ]
    for _, row in rows.iterrows():
        symbol = str(row["symbol"])[:8]
        lines.append(f"{int(row['rank']):>2} {symbol:<7} | {_fmt(row['predicted_open']):>9} | {_fmt(row['predicted_high']):>9} | {_fmt(row['predicted_low']):>9} | {_fmt(row['predicted_close']):>9}")
    lines.append("</pre>")
    return "\n".join(lines)


def send_morning() -> None:
    if not PREDICTIONS_FILE.exists():
        raise RuntimeError("Predictions file does not exist")
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["target_date"])
    target_date = predictions["target_date"].dropna().dt.normalize().max()
    if pd.isna(target_date):
        raise RuntimeError("No target dates found in predictions.csv")
    message = build_morning_message(predictions, target_date)
    if _already_sent(SENT_FILE, target_date, message):
        print(f"Morning Telegram already sent for {target_date.date()} with this exact message; skipping duplicate.")
        return
    _send(message)
    _record_sent(SENT_FILE, target_date, message)
    print(f"Morning Telegram sent for {target_date.date()}")


def _accuracy(rows: pd.DataFrame, field: str) -> float:
    err = pd.to_numeric(rows[f"{field}_abs_pct_error"], errors="coerce").dropna()
    return max(0.0, 100.0 - err.mean() * 100.0) if not err.empty else float("nan")


def _baseline_accuracy(rows: pd.DataFrame, field: str) -> float:
    err = pd.to_numeric(rows[f"baseline_{field}_abs_pct_error"], errors="coerce").dropna()
    return max(0.0, 100.0 - err.mean() * 100.0) if not err.empty else float("nan")


def _window(evals: pd.DataFrame, days: int) -> pd.DataFrame:
    end = evals["target_date"].max().normalize()
    start = end - pd.Timedelta(days=days - 1)
    return evals[evals["target_date"].between(start, end)]


def build_evening_message(evals: pd.DataFrame, target_date: pd.Timestamp) -> str:
    evals = evals.copy()
    evals["target_date"] = pd.to_datetime(evals["target_date"], errors="coerce").dt.normalize()
    rows = evals[evals["target_date"] == target_date.normalize()].sort_values("rank")
    if rows.empty:
        raise RuntimeError(f"No evaluations found for {target_date.date()}")
    metrics = {field: _accuracy(rows, field) for field in ["open", "high", "low", "close"]}
    overall = pd.Series(metrics, dtype="float64").mean()
    baseline_close = _baseline_accuracy(rows, "close")
    direction = pd.to_numeric(rows.get("close_direction_correct"), errors="coerce").mean() * 100 if "close_direction_correct" in rows else float("nan")
    lines = [
        "<b>STOCK PICKER</b>",
        f"{target_date:%d-%b-%Y} | EVENING",
        f"{len(rows)} predictions evaluated",
        "",
        "<pre>",
        "Index   | P/A             | Δ       | Err%",
        "------------------------------------------------",
    ]
    for _, row in rows.head(10).iterrows():
        diff = row["actual_close"] - row["predicted_close"]
        err = row["close_abs_pct_error"] * 100
        pa = f"{_fmt(row['predicted_close'])}/{_fmt(row['actual_close'])}"
        lines.append(f"{int(row['rank']):>2} {str(row['symbol'])[:7]:<7} | {pa:>15} | {diff:>7.2f} | {err:>5.2f}")
    lines += [
        "</pre>", "", "<b>MODEL ACCURACY</b>",
        f"Open     {_fmt(metrics['open'])}%", f"High     {_fmt(metrics['high'])}%", f"Low      {_fmt(metrics['low'])}%",
        f"Close    {_fmt(metrics['close'])}%", f"Overall  <b>{_fmt(overall)}%</b>", f"Direction {_fmt(direction)}%",
        f"Baseline Close {_fmt(baseline_close)}%", "", "<b>ROLLING CLOSE ACCURACY</b>",
    ]
    for days in [7, 30, 90]:
        window = _window(evals, days)
        lines.append(f"{days:>2}d       {_fmt(_accuracy(window, 'close'))}% | baseline {_fmt(_baseline_accuracy(window, 'close'))}% | direction {_fmt(pd.to_numeric(window.get('close_direction_correct'), errors='coerce').mean() * 100 if 'close_direction_correct' in window else float('nan'))}%")
    return "\n".join(lines)


def send_evening() -> None:
    if not EVALUATIONS_FILE.exists():
        print("No evaluations file yet; skipping evening Telegram report.")
        return
    evals = pd.read_csv(EVALUATIONS_FILE, parse_dates=["target_date"])
    if evals.empty:
        print("No evaluations yet; skipping evening Telegram report.")
        return
    target_date = evals["target_date"].dropna().dt.normalize().max()
    if pd.isna(target_date):
        print("No valid evaluation date; skipping evening Telegram report.")
        return
    message = build_evening_message(evals, target_date)
    if _already_sent(EVENING_SENT_FILE, target_date, message):
        print(f"Evening Telegram already sent for {target_date.date()} with this exact message; skipping duplicate.")
        return
    _send(message)
    _record_sent(EVENING_SENT_FILE, target_date, message)
    print(f"Evening Telegram sent for {target_date.date()}")


def build_paper_trading_message(trades: pd.DataFrame, target_date: pd.Timestamp) -> str:
    trades = trades.copy()
    trades["target_date"] = pd.to_datetime(trades["target_date"], errors="coerce").dt.normalize()
    rows = trades[trades["target_date"] == target_date.normalize()].sort_values("rank")
    if rows.empty:
        raise RuntimeError(f"No paper trades found for {target_date.date()}")
    traded = rows[rows["signal"] == "BUY"].copy()
    total_pnl = pd.to_numeric(traded["profit_loss"], errors="coerce").fillna(0).sum()
    win_rate = (traded["profit_loss"] > 0).mean() * 100 if not traded.empty else 0
    portfolio = pd.read_csv(PORTFOLIO_FILE) if PORTFOLIO_FILE.exists() else pd.DataFrame()
    latest_value = float(portfolio.iloc[-1]["portfolio_value"]) if not portfolio.empty else 100000 + total_pnl * 10
    lines = [
        "<b>PAPER TRADING</b>", f"{target_date:%d-%b-%Y} | LEARNED ENTRY/EXIT", "",
        "<pre>", "Stock     | Buy       | Sell      | P/L       | P/L%",
        "------------------------------------------------------",
    ]
    for _, row in rows.iterrows():
        if row["signal"] == "SKIP":
            lines.append(f"{str(row['symbol'])[:9]:<9} | SKIP      | -         | -         | -")
        else:
            lines.append(f"{str(row['symbol'])[:9]:<9} | {_fmt(row['entry_price']):>9} | {_fmt(row['exit_price']):>9} | {_fmt(row['profit_loss']):>9} | {_fmt(row['return_pct']):>6}%")
    lines += ["</pre>", "", f"Trades     {len(traded)}", f"Winners    {int((traded['profit_loss'] > 0).sum())}", f"Losers     {int((traded['profit_loss'] < 0).sum())}", f"Win rate   {_fmt(win_rate)}%", f"Net P/L    ₹{_fmt(total_pnl)}", f"Portfolio  ₹{latest_value:,.2f}"]
    return "\n".join(lines)


def send_paper_trading() -> None:
    if not PAPER_TRADES_FILE.exists():
        print("No paper trades yet; skipping paper trading Telegram report.")
        return
    trades = pd.read_csv(PAPER_TRADES_FILE, parse_dates=["target_date"])
    if trades.empty:
        print("No paper trades yet; skipping paper trading Telegram report.")
        return
    target_date = trades["target_date"].dropna().dt.normalize().max()
    if pd.isna(target_date):
        print("No valid paper trading date; skipping paper trading Telegram report.")
        return
    message = build_paper_trading_message(trades, target_date)
    if _already_sent(PAPER_SENT_FILE, target_date, message):
        print(f"Paper trading Telegram already sent for {target_date.date()}; skipping duplicate.")
        return
    _send(message)
    _record_sent(PAPER_SENT_FILE, target_date, message)
    print(f"Paper trading Telegram sent for {target_date.date()}")


if __name__ == "__main__":
    send_morning()
