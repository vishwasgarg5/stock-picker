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



def test_outcomes_preserve_separate_forecast_vintages_for_same_target(tmp_path, monkeypatch):
    import src.outcomes as outcomes

    predictions = pd.DataFrame([
        {"prediction_date": "2026-01-02", "target_date": "2026-01-05",
         "symbol": "aaa", "rank": 1, "score": 80, "base_close": 100,
         "predicted_close": 101},
        {"prediction_date": "2026-01-03", "target_date": "2026-01-05",
         "symbol": "AAA", "rank": 2, "score": 79, "base_close": 100,
         "predicted_close": 102},
    ])
    history = pd.DataFrame([
        {"date": "2026-04-06", "symbol": "AAA", "close": 110},
    ])
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    history.to_csv(tmp_path / "ohlcv.csv", index=False)
    monkeypatch.setattr(outcomes, "PREDICTIONS_FILE", tmp_path / "predictions.csv")
    monkeypatch.setattr(outcomes, "HISTORY_FILE", tmp_path / "ohlcv.csv")
    monkeypatch.setattr(outcomes, "OUTPUT_FILE", tmp_path / "prediction_outcomes.csv")

    result = outcomes.run_outcomes()

    assert len(result) == 2
    assert set(result["prediction_date"].dt.strftime("%Y-%m-%d")) == {
        "2026-01-02", "2026-01-03",
    }
    assert result["3m_status"].eq("AVAILABLE").all()


def test_selection_validation_scores_forecast_vintages_independently(tmp_path, monkeypatch):
    import src.selection_validation as validation

    symbols = [f"S{i}" for i in range(10)]
    candidates = []
    for prediction_date in ["2026-01-02", "2026-01-03"]:
        for rank, symbol in enumerate(symbols, start=1):
            candidates.append({
                "prediction_date": prediction_date,
                "target_date": "2026-01-05",
                "symbol": symbol,
                "base_close": 100.0,
                "predicted_close": 101.0 + rank / 100.0,
                "rank": rank,
                "selected": 1,
            })
    history = [
        {"date": "2026-01-05", "symbol": symbol, "close": 102.0}
        for symbol in symbols
    ]
    pd.DataFrame(candidates).to_csv(tmp_path / "prediction_candidates_history.csv", index=False)
    pd.DataFrame(history).to_csv(tmp_path / "ohlcv.csv", index=False)
    monkeypatch.setattr(validation, "CANDIDATES_FILE", tmp_path / "prediction_candidates_history.csv")
    monkeypatch.setattr(validation, "HISTORY_FILE", tmp_path / "ohlcv.csv")
    monkeypatch.setattr(validation, "OUTPUT_FILE", tmp_path / "selection_validation.csv")

    result = validation.run_selection_validation().iloc[0]

    assert result["sessions"] == 2
    assert result["candidate_rows"] == 20
    assert result["recent_error_promotion_evidence"] == False


def test_governance_steps_are_not_hardcoded_complete_without_evidence(tmp_path, monkeypatch):
    import src.model_governance as governance

    monkeypatch.setattr(governance, "DATA", tmp_path)
    monkeypatch.setattr(governance, "SOURCE_DIRS", [])
    result = governance.run_governance()

    assert result["governance_steps_total"] == 20
    assert result["governance_steps_with_evidence"] < 20
    assert result["governance_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["production_changes_authorized"] is False
    assert result["v2_remains_shadow_only"] is True
    assert result["feature_stability_status"] == "INSUFFICIENT_DATA"
    assert result["stable_features"] is None



def test_governance_stability_uses_latest_ten_sessions_not_file_order():
    from src.model_governance import _stock_stability

    rows = []
    for day in range(1, 13):
        rows.append({
            "symbol": "AAA",
            "target_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
            "close_direction_correct": 1 if day <= 10 else 0,
        })
    result = _stock_stability(pd.DataFrame(rows[::-1])).iloc[0]

    assert result["recent_accuracy_pct"] == 80.0


def test_governance_loss_streak_uses_chronological_trade_order(monkeypatch):
    import src.model_governance as governance

    trades = pd.DataFrame([
        {"target_date": "2026-01-01", "symbol": "A", "signal": "BUY", "profit_loss": -10.0},
        {"target_date": "2026-01-02", "symbol": "A", "signal": "BUY", "profit_loss": 5.0},
        {"target_date": "2026-01-03", "symbol": "A", "signal": "BUY", "profit_loss": -20.0},
        {"target_date": "2026-01-04", "symbol": "A", "signal": "BUY", "profit_loss": 5.0},
        {"target_date": "2026-01-05", "symbol": "A", "signal": "BUY", "profit_loss": -30.0},
    ])
    monkeypatch.setattr(governance, "_read", lambda name: trades.iloc[[4, 2, 0, 3, 1]].copy())
    result = governance._tail_risk()

    assert result["consecutive_losses"] == 1



def test_selection_validation_clears_stale_promotion_on_duplicate_vintage_keys(tmp_path, monkeypatch):
    import src.selection_validation as validation

    candidates = pd.DataFrame([
        {"prediction_date": "2026-01-02", "target_date": "2026-01-05",
         "symbol": "AAA", "rank": 1, "selected": 1,
         "base_close": 100, "predicted_close": 101},
        {"prediction_date": "2026-01-02", "target_date": "2026-01-05",
         "symbol": "AAA", "rank": 1, "selected": 1,
         "base_close": 100, "predicted_close": 101},
    ])
    candidates.to_csv(tmp_path / "candidates.csv", index=False)
    pd.DataFrame([{"date": "2026-01-05", "symbol": "AAA", "close": 102}]).to_csv(
        tmp_path / "history.csv", index=False
    )
    pd.DataFrame([{"recent_error_promotion_evidence": True}]).to_csv(
        tmp_path / "validation.csv", index=False
    )
    monkeypatch.setattr(validation, "CANDIDATES_FILE", tmp_path / "candidates.csv")
    monkeypatch.setattr(validation, "HISTORY_FILE", tmp_path / "history.csv")
    monkeypatch.setattr(validation, "OUTPUT_FILE", tmp_path / "validation.csv")

    result = validation.run_selection_validation().iloc[0]

    assert result["validation_status"] == "blocked_duplicate_candidate_keys"
    assert result["recent_error_promotion_evidence"] == False
    assert result["confidence_promotion_evidence"] == False



def test_v1_challenger_selectors_require_explicit_human_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    readiness = tmp_path / "phase114_promotion_readiness_summary.json"

    assert pipeline._production_selector_authorized() is False

    readiness.write_text('{"production_changes_authorized": "true"}', encoding="utf-8")
    assert pipeline._production_selector_authorized() is False

    readiness.write_text('{"production_changes_authorized": true}', encoding="utf-8")
    assert pipeline._production_selector_authorized() is True
