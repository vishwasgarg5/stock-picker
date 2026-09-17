from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = ROOT / "data" / "predictions.csv"


def _fmt(value: object) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def build_message(predictions: pd.DataFrame, target_date: pd.Timestamp) -> str:
    rows = predictions[predictions["target_date"].dt.normalize() == target_date.normalize()].sort_values("rank").head(10)
    if rows.empty:
        raise RuntimeError(f"No predictions found for {target_date.date()}")

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


def send_telegram(target_date: pd.Timestamp) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured")
    if not PREDICTIONS_FILE.exists():
        raise RuntimeError("Predictions file does not exist")

    predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=["target_date"])
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
    print(f"Telegram message sent for {target_date.date()}")
