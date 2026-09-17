from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS_FILE = DATA / "predictions.csv"
SENT_FILE = DATA / "telegram_sent.csv"


def _fmt(value: object) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def build_message(predictions: pd.DataFrame, target_date: pd.Timestamp) -> str:
    rows = predictions[predictions["target_date"].dt.normalize() == target_date.normalize()].sort_values("rank").head(10)
    if len(rows) < 10:
        raise RuntimeError(f"Expected 10 predictions for {target_date.date()}, found {len(rows)}")

    lines = [
        "<b>🚀 STOCK PICKER — MORNING</b>",
        f"📅 Target: <b>{target_date:%d-%b-%Y}</b>",
        "📈 Universe: Nifty Midcap 150",
        "",
        "<pre>",
        "#  Stock     Open     High      Low    Close",
        "────────────────────────────────────────────",
    ]
    for _, row in rows.iterrows():
        symbol = str(row["symbol"])[:8]
        lines.append(
            f"{int(row['rank']):<3} {symbol:<8} {_fmt(row['predicted_open']):>8} {_fmt(row['predicted_high']):>8} {_fmt(row['predicted_low']):>8} {_fmt(row['predicted_close']):>8}"
        )
    lines += [
        "</pre>",
        "🤖 <i>Automated model prediction</i>",
        "⚠️ <i>For informational purposes only.</i>",
    ]
    return "\n".join(lines)


def _latest_target(predictions: pd.DataFrame) -> pd.Timestamp:
    if "target_date" not in predictions.columns:
        raise RuntimeError("predictions.csv has no target_date column")
    dates = predictions["target_date"].dropna().dt.normalize()
    if dates.empty:
        raise RuntimeError("No target dates found in predictions.csv")
    return dates.max()


def _already_sent(target_date: pd.Timestamp) -> bool:
    if not SENT_FILE.exists():
        return False
    sent = pd.read_csv(SENT_FILE)
    if "target_date" not in sent.columns:
        return False
    dates = pd.to_datetime(sent["target_date"], errors="coerce").dt.normalize()
    return dates.eq(target_date.normalize()).any()


def _record_sent(target_date: pd.Timestamp) -> None:
    if SENT_FILE.exists():
        sent = pd.read_csv(SENT_FILE)
    else:
        sent = pd.DataFrame(columns=["target_date"])
    sent = pd.concat([sent, pd.DataFrame({"target_date": [target_date.date().isoformat()]})], ignore_index=True)
    sent.drop_duplicates("target_date", keep="first").to_csv(SENT_FILE, index=False)


def send_telegram(target_date: pd.Timestamp | None = None) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured")
    if not PREDICTIONS_FILE.exists():
        raise RuntimeError("Predictions file does not exist")

    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["target_date"])
    target_date = _latest_target(predictions) if target_date is None else pd.Timestamp(target_date).normalize()
    if _already_sent(target_date):
        print(f"Telegram already sent for {target_date.date()}; skipping duplicate.")
        return

    message = build_message(predictions, target_date)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")
    _record_sent(target_date)
    print(f"Telegram message sent for {target_date.date()}")


if __name__ == "__main__":
    send_telegram()
