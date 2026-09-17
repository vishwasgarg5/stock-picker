from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
EVALUATIONS_FILE = DATA / "evaluations.csv"
SENT_FILE = DATA / "telegram_sent.csv"
EVENING_SENT_FILE = DATA / "telegram_evening_sent.csv"


def _fmt(value: object) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


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


def _already_sent(path: Path, target_date: pd.Timestamp) -> bool:
    if not path.exists():
        return False
    sent = pd.read_csv(path)
    if "target_date" not in sent.columns:
        return False
    return pd.to_datetime(sent["target_date"], errors="coerce").dt.normalize().eq(target_date.normalize()).any()


def _record_sent(path: Path, target_date: pd.Timestamp) -> None:
    sent = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["target_date"])
    sent = pd.concat([sent, pd.DataFrame({"target_date": [target_date.date().isoformat()]})], ignore_index=True)
    sent.drop_duplicates("target_date", keep="first").to_csv(path, index=False)


def build_morning_message(predictions: pd.DataFrame, target_date: pd.Timestamp) -> str:
    rows = predictions[predictions["target_date"].dt.normalize() == target_date.normalize()].sort_values("rank").head(10)
    if len(rows) < 10:
        raise RuntimeError(f"Expected 10 predictions for {target_date.date()}, found {len(rows)}")
    lines = [
        "<b>🚀 STOCK PICKER — MORNING</b>",
        f"📅 Target: <b>{target_date:%d-%b-%Y}</b>",
        "📈 Universe: Nifty Midcap 150", "", "<pre>",
        "#  Stock     Open     High      Low    Close",
        "────────────────────────────────────────────",
    ]
    for _, row in rows.iterrows():
        symbol = str(row["symbol"])[:8]
        lines.append(f"{int(row['rank']):<3} {symbol:<8} {_fmt(row['predicted_open']):>8} {_fmt(row['predicted_high']):>8} {_fmt(row['predicted_low']):>8} {_fmt(row['predicted_close']):>8}")
    lines += ["</pre>", "🤖 <i>Automated model prediction</i>", "⚠️ <i>For informational purposes only.</i>"]
    return "\n".join(lines)


def send_morning() -> None:
    if not PREDICTIONS_FILE.exists():
        raise RuntimeError("Predictions file does not exist")
    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["target_date"])
    target_date = predictions["target_date"].dropna().dt.normalize().max()
    if pd.isna(target_date):
        raise RuntimeError("No target dates found in predictions.csv")
    if _already_sent(SENT_FILE, target_date):
        print(f"Morning Telegram already sent for {target_date.date()}; skipping duplicate.")
        return
    _send(build_morning_message(predictions, target_date))
    _record_sent(SENT_FILE, target_date)
    print(f"Morning Telegram sent for {target_date.date()}")


def build_evening_message(evals: pd.DataFrame, target_date: pd.Timestamp) -> str:
    rows = evals[evals["target_date"].dt.normalize() == target_date.normalize()].sort_values("rank")
    if rows.empty:
        raise RuntimeError(f"No evaluations found for {target_date.date()}")

    metrics = {}
    for field in ["open", "high", "low", "close"]:
        err = rows[f"{field}_abs_pct_error"].dropna()
        metrics[field] = max(0.0, 100.0 - err.mean() * 100.0) if not err.empty else float("nan")

    direction_correct = ((rows["predicted_close"] - rows["base_close"]) * (rows["actual_close"] - rows["base_close"]) > 0).sum() if "base_close" in rows.columns else None
    lines = [
        "<b>🌙 STOCK PICKER — EVENING</b>",
        f"📅 Session: <b>{target_date:%d-%b-%Y}</b>",
        f"📊 Predictions evaluated: <b>{len(rows)}</b>", "", "<pre>",
        "#  Stock    P.Close A.Close     Δ   Err%",
        "────────────────────────────────────────",
    ]
    for _, row in rows.head(10).iterrows():
        diff = row["actual_close"] - row["predicted_close"]
        err = row["close_abs_pct_error"] * 100
        lines.append(f"{int(row['rank']):<3} {str(row['symbol'])[:8]:<8} {_fmt(row['predicted_close']):>8} {_fmt(row['actual_close']):>8} {diff:>7.2f} {err:>6.2f}")
    lines += [
        "</pre>", "", "<b>📈 MODEL ACCURACY</b>",
        f"Open   : {_fmt(metrics['open'])}%",
        f"High   : {_fmt(metrics['high'])}%",
        f"Low    : {_fmt(metrics['low'])}%",
        f"Close  : {_fmt(metrics['close'])}%",
        f"Overall: {_fmt(pd.Series(metrics).mean())}%",
    ]
    if direction_correct is not None:
        lines.append(f"🎯 Close direction: {direction_correct}/{len(rows)}")
    lines += ["🤖 <i>Models retrained after evaluation</i>", "⚠️ <i>Historical accuracy does not guarantee future results.</i>"]
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
    if _already_sent(EVENING_SENT_FILE, target_date):
        print(f"Evening Telegram already sent for {target_date.date()}; skipping duplicate.")
        return
    _send(build_evening_message(evals, target_date))
    _record_sent(EVENING_SENT_FILE, target_date)
    print(f"Evening Telegram sent for {target_date.date()}")


if __name__ == "__main__":
    send_morning()
