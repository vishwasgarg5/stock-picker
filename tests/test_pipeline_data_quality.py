import pandas as pd

from src import pipeline
from src.pipeline import _merge_candidate_history, _select_completed_market_date, validate_data_quality


def _rows(date, symbols, invalid=False):
    rows = []
    for i, symbol in enumerate(symbols):
        close = 100.0 + i
        high = close + 1.0
        low = close - 1.0
        if invalid:
            high, low = close - 2.0, close + 2.0
        rows.append({
            "date": date,
            "symbol": symbol,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        })
    return rows


def test_partial_current_session_falls_back_to_previous_clean_session():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    today = pd.Timestamp.now().normalize()
    hist = pd.DataFrame(_rows(today, symbols, invalid=True) + _rows(yesterday, symbols))
    assert _select_completed_market_date(hist, symbols) == yesterday


def test_completed_session_quality_gate_uses_clean_session():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    session = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    hist = pd.DataFrame(_rows(session, symbols))
    result = validate_data_quality(hist, symbols, "test")
    assert result["latest_date"] == session
    assert result["coverage"] == 1.0



def test_non_positive_open_and_low_prices_are_rejected():
    symbol = "AAA"
    fallback = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
    newer = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    clean = pd.DataFrame(_rows(fallback, [symbol]))
    corrupt = pd.DataFrame(_rows(newer, [symbol]))
    corrupt.loc[0, "open"] = 0.0
    corrupt.loc[0, "low"] = 0.0

    hist = pd.concat([clean, corrupt], ignore_index=True)

    assert _select_completed_market_date(hist, [symbol]) == fallback


def test_non_finite_ohlc_prices_are_rejected():
    symbol = "AAA"
    fallback = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)
    newer = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    clean = pd.DataFrame(_rows(fallback, [symbol]))
    corrupt = pd.DataFrame(_rows(newer, [symbol]))
    corrupt.loc[0, "high"] = float("inf")

    hist = pd.concat([clean, corrupt], ignore_index=True)

    assert _select_completed_market_date(hist, [symbol]) == fallback



def test_candidate_history_preserves_repeated_target_date_across_prediction_sessions():
    existing = pd.DataFrame([
        {"prediction_date": "2026-11-12", "target_date": "2026-11-16", "symbol": "MCX", "score": 70.0},
    ])
    current = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "MCX", "score": 80.0},
    ])

    result = _merge_candidate_history(existing, current)

    assert len(result) == 2
    assert set(pd.to_datetime(result["prediction_date"]).dt.strftime("%Y-%m-%d")) == {
        "2026-11-12", "2026-11-13",
    }
    assert result["symbol"].tolist() == ["MCX", "MCX"]


def test_candidate_history_rerun_refreshes_same_prediction_session():
    existing = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "MCX", "score": 70.0},
    ])
    rerun = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "MCX", "score": 85.0},
    ])

    result = _merge_candidate_history(existing, rerun)

    assert len(result) == 1
    assert result.iloc[0]["score"] == 85.0



def test_existing_prediction_refreshes_candidate_history_without_replacing_prediction(monkeypatch):
    target_date = pd.Timestamp("2026-11-16")
    existing = pd.DataFrame([
        {
            "prediction_date": pd.Timestamp("2026-11-13"),
            "target_date": target_date,
            "symbol": f"STOCK{i}",
            "rank": i + 1,
        }
        for i in range(10)
    ])
    calls = []

    monkeypatch.setattr(
        pipeline,
        "_validate_prediction_session",
        lambda frame, target: calls.append(("validate", len(frame), target)),
    )
    monkeypatch.setattr(
        pipeline,
        "predict_top10",
        lambda hist, ranking, target: calls.append(("refresh", target)),
    )

    pipeline._save_next_session_prediction(
        pd.DataFrame({"symbol": ["STOCK0"]}),
        pd.DataFrame({"symbol": ["STOCK0"]}),
        target_date,
        existing,
    )

    assert calls == [
        ("validate", 10, target_date),
        ("refresh", target_date),
    ]


def test_daily_workflow_serializes_manual_and_scheduled_runs():
    from pathlib import Path

    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    concurrency = workflow.split("concurrency:", 1)[1].split("\nenv:", 1)[0]

    assert "group: stock-picker-daily-${{ github.ref }}" in concurrency
    assert "github.event_name" not in concurrency
    assert "cancel-in-progress: false" in concurrency



def test_yahoo_ticker_uses_rr_series_alias_for_bagmane():
    assert pipeline._yahoo_ticker("BAGMANE") == "BAGMANE-RR.NS"
    assert pipeline._yahoo_ticker("RELIANCE") == "RELIANCE.NS"


def test_paper_trade_source_records_net_accounting_and_explicit_session_close():
    rows = pd.DataFrame([
        {
            "symbol": "AAA", "target_date": "2026-11-16", "signal": "BUY",
            "quantity": 10.0, "entry_price": 100.0, "exit_price": 110.0,
            "no_trade_reason": "",
        },
        {
            "symbol": "BBB", "target_date": "2026-11-16", "signal": "BUY",
            "quantity": 0.0, "entry_price": 100.0, "exit_price": 105.0,
            "no_trade_reason": "",
        },
    ])

    result = pipeline._apply_paper_trade_accounting(rows)
    traded = result.set_index("symbol").loc["AAA"]
    skipped = result.set_index("symbol").loc["BBB"]

    assert traded["shares"] == 10
    assert traded["allocated_capital"] == 1000.0
    assert traded["gross_profit_loss"] == 100.0
    assert traded["costs"] == 1.0
    assert traded["profit_loss"] == 99.0
    assert traded["return_pct"] == 9.9
    assert traded["exit_reason"] == "session_close"
    assert traded["exit_date"] == "2026-11-16"
    assert traded["ledger_schema_version"] == 3
    assert traded["cost_model"] == "fixed_round_trip_0.10pct_of_entry_notional"

    assert skipped["signal"] == "SKIP"
    assert skipped["no_trade_reason"] == "minimum_one_share_not_affordable"
    assert pd.isna(skipped["shares"])
    assert skipped["exit_reason"] == ""
