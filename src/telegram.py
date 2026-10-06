from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
CANDIDATES_FILE = DATA / "prediction_candidates.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"
PAPER_TRADES_FILE = DATA / "paper_trades.csv"
PORTFOLIO_FILE = DATA / "portfolio_daily.csv"
UNIVERSE_FILE = DATA / "universe.csv"
NEW_LISTINGS_FILE = DATA / "new_listings.csv"
OHLCV_FILE = DATA / "ohlcv.csv"
SENT_FILE = DATA / "telegram_sent.csv"
EVENING_SENT_FILE = DATA / "telegram_evening_sent.csv"
PAPER_SENT_FILE = DATA / "telegram_paper_sent.csv"
V2_TRADES_FILE = DATA / "paper_trades_v2.csv"
V2_PORTFOLIO_FILE = DATA / "portfolio_v2_daily.csv"
V2_SENT_FILE = DATA / "telegram_paper_v2_sent.csv"
PAPER_CAPITAL = 100000.0
PAPER_TRADE_TOP_N = 5


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


def _already_sent(path: Path, target_date: pd.Timestamp, message: str, force: bool = False) -> bool:
    if force:
        return False
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


def _previous_closes(target_date: pd.Timestamp, symbols: pd.Series) -> dict[str, float]:
    if not OHLCV_FILE.exists():
        raise RuntimeError("OHLCV file does not exist")
    columns = ["symbol", "date", "close"]
    ohlcv = pd.read_csv(OHLCV_FILE, usecols=columns, parse_dates=["date"])
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.strip()
    ohlcv["close"] = pd.to_numeric(ohlcv["close"], errors="coerce")
    target = target_date.normalize()
    previous = ohlcv[ohlcv["date"].dt.normalize() < target].dropna(subset=["close"])
    previous = previous[previous["symbol"].isin(symbols.astype(str).str.strip())]
    if previous.empty:
        raise RuntimeError(f"No prior closes found before {target_date.date()}")
    previous = previous.sort_values(["symbol", "date"]).drop_duplicates("symbol", keep="last")
    return dict(zip(previous["symbol"], previous["close"]))


def _open_gap_pct(predicted_open: object, previous_close: object) -> float:
    try:
        predicted = float(predicted_open)
        previous = float(previous_close)
        if previous == 0:
            return float("nan")
        return (predicted - previous) / previous * 100.0
    except (TypeError, ValueError):
        return float("nan")


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "-"


def build_morning_message(predictions: pd.DataFrame, target_date: pd.Timestamp) -> str:
    rows = predictions[predictions["target_date"].dt.normalize() == target_date.normalize()].sort_values("rank").head(10)
    if len(rows) < 10:
        raise RuntimeError(f"Expected 10 predictions for {target_date.date()}, found {len(rows)}")
    total_stocks = _universe_count()
    previous_closes = _previous_closes(target_date, rows["symbol"])
    selection_method = "-"
    rejected = pd.DataFrame()
    if CANDIDATES_FILE.exists():
        try:
            candidates = pd.read_csv(CANDIDATES_FILE)
            candidates["target_date"] = pd.to_datetime(candidates["target_date"], errors="coerce").dt.normalize()
            candidates = candidates[candidates["target_date"] == target_date.normalize()].copy()
            if not candidates.empty:
                selection_method = str(candidates["selection_method"].dropna().iloc[0]) if candidates["selection_method"].notna().any() else "-"
                rejected = candidates[candidates["selected"].astype(int).eq(0)].sort_values("rank").head(3)
        except Exception:
            pass
    lines = [
        "<b>STOCK PICKER</b>",
        f"{target_date:%d-%b-%Y} | TOP 10 / {total_stocks}",
        f"<b>Selection:</b> {selection_method}",
        "",
        "<pre>",
        "Index   | Open      | High      | Low       | Close     | O→PC %",
        "---------------------------------------------------------------",
    ]
    for _, row in rows.iterrows():
        symbol = str(row["symbol"]).strip()
        previous_close = previous_closes.get(symbol, float("nan"))
        gap_pct = _open_gap_pct(row["predicted_open"], previous_close)
        lines.append(
            f"{int(row['rank']):>2} {symbol[:7]:<7} | "
            f"{_fmt(row['predicted_open']):>9} | {_fmt(row['predicted_high']):>9} | "
            f"{_fmt(row['predicted_low']):>9} | {_fmt(row['predicted_close']):>9} | {_fmt_pct(gap_pct):>7}"
        )
    lines.append("</pre>")

    if NEW_LISTINGS_FILE.exists():
        try:
            listings = pd.read_csv(NEW_LISTINGS_FILE)
            if not listings.empty and {"symbol", "status", "calendar_age"}.issubset(listings.columns):
                eligible = listings[listings["status"].isin(["ML_CANDIDATE_REDUCED_CONFIDENCE", "NORMAL"])].copy()
                if not eligible.empty:
                    names = ", ".join(
                        f"{str(r['symbol']).strip()} ({int(float(r['calendar_age']))}d)"
                        for _, r in eligible.head(5).iterrows()
                    )
                    lines += ["", "<b>New-listing candidates</b>", names]
        except Exception:
            pass

    if not rejected.empty:
        lines += ["", "<b>Nearest rejected</b>"]
        lines.append(", ".join(f"{str(row['symbol']).strip()} (rank {int(row['rank'])})" for _, row in rejected.iterrows()))
    return "\n".join(lines)


def _force_telegram() -> bool:
    return os.environ.get("FORCE_TELEGRAM", "").strip().lower() in {"1", "true", "yes"}


def send_morning() -> None:
    if not PREDICTIONS_FILE.exists():
        raise RuntimeError("Predictions file does not exist")
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["target_date"])
    target_date = predictions["target_date"].dropna().dt.normalize().max()
    if pd.isna(target_date):
        raise RuntimeError("No target dates found in predictions.csv")
    message = build_morning_message(predictions, target_date)
    if _already_sent(SENT_FILE, target_date, message, force=_force_telegram()):
        print(f"Morning Telegram already sent for {target_date.date()} with this exact message; skipping duplicate.")
        return
    _send(message)
    _record_sent(SENT_FILE, target_date, message)
    print(f"Morning Telegram sent for {target_date.date()}")


def _mape(rows: pd.DataFrame, field: str) -> float:
    err = pd.to_numeric(rows[f"{field}_abs_pct_error"], errors="coerce").dropna()
    return err.mean() * 100.0 if not err.empty else float("nan")


def _baseline_mape(rows: pd.DataFrame, field: str) -> float:
    err = pd.to_numeric(rows[f"baseline_{field}_abs_pct_error"], errors="coerce").dropna()
    return err.mean() * 100.0 if not err.empty else float("nan")


def _window(evals: pd.DataFrame, days: int) -> pd.DataFrame:
    end = evals["target_date"].max().normalize()
    start = end - pd.Timedelta(days=days - 1)
    return evals[evals["target_date"].between(start, end)]


def _pct_diff(predicted: object, actual: object) -> float:
    try:
        predicted_value = float(predicted)
        actual_value = float(actual)
        if predicted_value == 0:
            return float("nan")
        return (predicted_value - actual_value) / predicted_value * 100.0
    except (TypeError, ValueError):
        return float("nan")


def build_evening_message(evals: pd.DataFrame, target_date: pd.Timestamp) -> str:
    evals = evals.copy()
    evals["target_date"] = pd.to_datetime(evals["target_date"], errors="coerce").dt.normalize()
    rows = evals[evals["target_date"] == target_date.normalize()].sort_values("rank").head(10)
    if rows.empty:
        raise RuntimeError(f"No evaluations found for {target_date.date()}")

    metrics = {field: _mape(rows, field) for field in ["open", "high", "low", "close"]}
    overall = pd.Series(metrics, dtype="float64").mean()
    baseline_close = _baseline_mape(rows, "close")
    direction = pd.to_numeric(rows.get("close_direction_correct"), errors="coerce").mean() * 100 if "close_direction_correct" in rows else float("nan")

    lines = [
        "<b>📊 STOCK PICKER — MODEL VALIDATION</b>",
        f"<b>{target_date:%d-%b-%Y}</b> | {len(rows)} predictions evaluated",
        "",
        "<b>OHLC: PREDICTED → ACTUAL</b>",
    ]
    for _, row in rows.iterrows():
        symbol = str(row["symbol"]).strip()
        lines += [
            "",
            f"<b>{int(row['rank'])}. {symbol}</b>",
            f"Open  {_fmt(row['predicted_open'])} → {_fmt(row['actual_open'])}  ({_fmt(row['actual_open'] - row['predicted_open'])})",
            f"High  {_fmt(row['predicted_high'])} → {_fmt(row['actual_high'])}  ({_fmt(row['actual_high'] - row['predicted_high'])})",
            f"Low   {_fmt(row['predicted_low'])} → {_fmt(row['actual_low'])}  ({_fmt(row['actual_low'] - row['predicted_low'])})",
            f"Close {_fmt(row['predicted_close'])} → {_fmt(row['actual_close'])}  ({_fmt(row['actual_close'] - row['predicted_close'])})",
        ]

    lines += [
        "",
        "<b>MODEL ERROR</b>",
        f"Open       {_fmt(metrics['open'])}%",
        f"High       {_fmt(metrics['high'])}%",
        f"Low        {_fmt(metrics['low'])}%",
        f"Close      {_fmt(metrics['close'])}%",
        f"<b>Overall    {_fmt(overall)}%</b>",
        f"Direction  {_fmt(direction)}%",
        f"Baseline C {_fmt(baseline_close)}%",
        "",
        "<b>ROLLING CLOSE MAPE</b>",
    ]
    for days in [7, 30, 90]:
        window = _window(evals, days)
        roll_direction = pd.to_numeric(window.get("close_direction_correct"), errors="coerce").mean() * 100 if "close_direction_correct" in window else float("nan")
        lines.append(
            f"{days}d  Model {_fmt(_mape(window, 'close'))}% | Baseline {_fmt(_baseline_mape(window, 'close'))}% | Dir {_fmt(roll_direction)}%"
        )
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
    if _already_sent(EVENING_SENT_FILE, target_date, message, force=_force_telegram()):
        print(f"Evening Telegram already sent for {target_date.date()} with this exact message; skipping duplicate.")
        return
    _send(message)
    _record_sent(EVENING_SENT_FILE, target_date, message)
    print(f"Evening Telegram sent for {target_date.date()}")


def build_paper_trading_message(trades: pd.DataFrame, target_date: pd.Timestamp) -> str:
    """Build mobile-friendly completed result + next-day paper portfolio."""
    trades = trades.copy()
    trades["target_date"] = pd.to_datetime(trades["target_date"], errors="coerce").dt.normalize()
    rows = trades[trades["target_date"] == target_date.normalize()].sort_values("rank")
    if rows.empty:
        raise RuntimeError(f"No paper trades found for {target_date.date()}")

    traded = rows[rows["signal"] == "BUY"].copy()
    total_pnl = pd.to_numeric(traded["profit_loss"], errors="coerce").fillna(0).sum()
    total_invested = pd.to_numeric(traded["position_value"], errors="coerce").fillna(0).sum()
    portfolio = pd.read_csv(PORTFOLIO_FILE) if PORTFOLIO_FILE.exists() else pd.DataFrame()
    latest_value = float(portfolio.iloc[-1]["portfolio_value"]) if not portfolio.empty else PAPER_CAPITAL + total_pnl
    actual_return = (total_pnl / PAPER_CAPITAL * 100.0) if PAPER_CAPITAL else 0.0

    lines = [
        "<b>💼 PAPER TRADING — ACTUAL RESULT</b>",
        f"<b>{target_date:%d-%b-%Y}</b> | Previous Top-5",
    ]
    for _, row in traded.iterrows():
        qty = int(row["quantity"]) if "quantity" in row.index and pd.notna(row["quantity"]) else 0
        pnl = float(row["profit_loss"])
        ret = float(row["return_pct"])
        sign = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        lines += [
            "",
            f"{sign} <b>{str(row['symbol']).strip()}</b>  × {qty}",
            f"Entry {_fmt(row['entry_price'])} → Exit {_fmt(row['exit_price'])}",
            f"P/L ₹{pnl:+,.2f}  ({ret:+.2f}%)",
        ]

    lines += [
        "",
        "──────────────",
        f"<b>Trades:</b> {len(traded)}  |  🟢 {int((traded['profit_loss'] > 0).sum())}  |  🔴 {int((traded['profit_loss'] < 0).sum())}",
        f"<b>Total P/L:</b> ₹{total_pnl:+,.2f}",
        f"<b>Portfolio return:</b> {actual_return:+.2f}%",
        f"<b>Portfolio value:</b> ₹{latest_value:,.2f}",
        "",
    ]

    next_date = pd.to_datetime(target_date).normalize()
    if PREDICTIONS_FILE.exists():
        predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["prediction_date", "target_date"])
        predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
        future = predictions[predictions["target_date"] > next_date].sort_values(["target_date", "rank"])
        if not future.empty:
            next_target = future["target_date"].min()
            nxt = future[future["target_date"] == next_target].sort_values("rank").head(5).copy()
            allocation = PAPER_CAPITAL / PAPER_TRADE_TOP_N
            nxt["reference_price"] = pd.to_numeric(nxt["base_close"], errors="coerce")
            nxt["qty"] = np.floor(allocation / nxt["reference_price"]).fillna(0).astype(int)
            nxt["used_capital"] = nxt["qty"] * nxt["reference_price"]
            cash_left = PAPER_CAPITAL - nxt["used_capital"].sum()

            lines += [
                "<b>📈 NEXT-DAY PAPER PORTFOLIO</b>",
                f"<b>{next_target:%d-%b-%Y}</b> | Top-5",
            ]
            for _, row in nxt.iterrows():
                lines += [
                    "",
                    f"<b>#{int(row['rank'])} {str(row['symbol']).strip()}</b>",
                    f"Qty {int(row['qty'])}  |  Ref ₹{_fmt(row['reference_price'])}",
                    f"Capital used ₹{row['used_capital']:,.0f}",
                ]
            lines += [
                "",
                "──────────────",
                f"<b>Capital:</b> ₹{PAPER_CAPITAL:,.0f}",
                f"<b>Used:</b> ₹{nxt['used_capital'].sum():,.0f}",
                f"<b>Cash:</b> ₹{cash_left:,.0f}",
                "",
                "Qty = floor(₹20,000 / reference price)",
                "Reference = latest completed close; entry = next-day actual open",
            ]
        else:
            lines += ["<b>📈 NEXT-DAY PAPER PORTFOLIO</b>", "No future prediction available yet."]
    return "\n".join(lines)


def build_paper_trading_v2_message(trades: pd.DataFrame, target_date: pd.Timestamp) -> str:
    trades = trades.copy()
    trades["target_date"] = pd.to_datetime(trades["target_date"], errors="coerce").dt.normalize()
    rows = trades[trades["target_date"] == target_date.normalize()].sort_values("rank")
    if rows.empty:
        raise RuntimeError(f"No V2 paper trades found for {target_date.date()}")
    buys = rows[rows["signal"].eq("BUY")].copy()
    pnl = pd.to_numeric(buys["profit_loss"], errors="coerce").fillna(0.0)
    portfolio = pd.read_csv(V2_PORTFOLIO_FILE) if V2_PORTFOLIO_FILE.exists() else pd.DataFrame()
    value = float(portfolio.iloc[-1]["portfolio_value"]) if not portfolio.empty else PAPER_CAPITAL + float(pnl.sum())
    reasons = rows[rows["signal"].ne("BUY")]["no_trade_reason"].value_counts().head(5) if "no_trade_reason" in rows else pd.Series(dtype=int)
    lines = [
        "<b>🛡️ PAPER TRADING V2 — RISK GATE</b>",
        f"<b>{target_date:%d-%b-%Y}</b> | {len(buys)} BUY / {len(rows)-len(buys)} NO_TRADE",
    ]
    for _, row in buys.iterrows():
        lines += [
            "",
            f"🟢 <b>{str(row['symbol']).strip()}</b> × {int(row['quantity'])}",
            f"Entry {_fmt(row['entry_price'])} → Exit {_fmt(row['exit_price'])}",
            f"P/L ₹{float(row['profit_loss']):+,.2f} | R/R {_fmt(row['risk_reward'])}",
        ]
    if not buys.empty:
        lines += ["", f"<b>V2 P/L:</b> ₹{pnl.sum():+,.2f}"]
    if not reasons.empty:
        lines += ["", "<b>NO_TRADE reasons</b>"]
        lines += [f"{idx}: {int(val)}" for idx, val in reasons.items()]
    lines += ["", f"<b>V2 Portfolio:</b> ₹{value:,.2f}", "V1 remains champion until governor promotion gates pass."]
    return "\n".join(lines)

def send_paper_trading_v2() -> None:
    if not V2_TRADES_FILE.exists():
        print("No V2 paper trades yet; skipping V2 Telegram report.")
        return
    trades = pd.read_csv(V2_TRADES_FILE, parse_dates=["target_date"])
    if trades.empty:
        print("No V2 paper trades yet; skipping V2 Telegram report.")
        return
    target_date = trades["target_date"].dropna().dt.normalize().max()
    if pd.isna(target_date):
        return
    message = build_paper_trading_v2_message(trades, target_date)
    if _already_sent(V2_SENT_FILE, target_date, message, force=_force_telegram()):
        print(f"V2 Telegram already sent for {target_date.date()}; skipping duplicate.")
        return
    _send(message)
    _record_sent(V2_SENT_FILE, target_date, message)
    print(f"V2 Telegram sent for {target_date.date()}")

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
    if _already_sent(PAPER_SENT_FILE, target_date, message, force=_force_telegram()):
        print(f"Paper trading Telegram already sent for {target_date.date()}; skipping duplicate.")
        return
    _send(message)
    _record_sent(PAPER_SENT_FILE, target_date, message)
    print(f"Paper trading Telegram sent for {target_date.date()}")


if __name__ == "__main__":
    send_morning()
