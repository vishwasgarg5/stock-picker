from __future__ import annotations

import pandas as pd
import pytest

from src import telegram


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


def test_send_requires_telegram_message_id_and_returns_receipt(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setattr(
        telegram.requests, "post",
        lambda *args, **kwargs: _Response({"ok": True, "result": {"message_id": 123, "date": 1791820800}}),
    )

    receipt = telegram._send("test message")

    assert receipt["delivery_status"] == "TELEGRAM_API_CONFIRMED"
    assert receipt["telegram_message_id"] == "123"
    assert receipt["telegram_date_utc"] == "1791820800"


def test_send_rejects_ok_response_without_message_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setattr(telegram.requests, "post", lambda *args, **kwargs: _Response({"ok": True, "result": {}}))

    with pytest.raises(RuntimeError, match="message_id"):
        telegram._send("test message")


def test_record_sent_persists_receipt_metadata(tmp_path):
    path = tmp_path / "telegram_sent.csv"
    receipt = {
        "delivery_status": "TELEGRAM_API_CONFIRMED",
        "telegram_message_id": "456",
        "telegram_date_utc": "1791820800",
    }

    telegram._record_sent(path, pd.Timestamp("2026-10-12"), "hello", receipt)
    saved = pd.read_csv(path)

    assert saved.loc[0, "delivery_status"] == "TELEGRAM_API_CONFIRMED"
    assert str(saved.loc[0, "telegram_message_id"]) == "456"
    assert str(saved.loc[0, "telegram_date_utc"]) == "1791820800"
