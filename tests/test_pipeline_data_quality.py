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
    assert traded["accounting_reconciliation_status"] == "RECONCILED_SCHEMA_V3"
    assert traded["exit_reason_record_status"] == "RECORDED"

    assert skipped["signal"] == "SKIP"
    assert skipped["no_trade_reason"] == "minimum_one_share_not_affordable"
    assert pd.isna(skipped["shares"])
    assert skipped["exit_reason"] == ""
    assert skipped["accounting_reconciliation_status"] == "NOT_APPLICABLE"
    assert skipped["exit_reason_record_status"] == "NOT_APPLICABLE"



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


def test_selection_validation_reports_missing_candidate_vintages_and_empty_market_history(tmp_path, monkeypatch):
    import json
    import src.selection_validation as validation

    prediction_dates = pd.date_range("2026-01-01", periods=13, freq="B")
    predictions = pd.DataFrame({
        "prediction_date": prediction_dates,
        "target_date": prediction_dates + pd.offsets.BDay(1),
        "symbol": ["AAA"] * len(prediction_dates),
    })
    candidates = pd.DataFrame([{
        "prediction_date": prediction_dates[0],
        "target_date": prediction_dates[0] + pd.offsets.BDay(1),
        "symbol": "AAA",
        "base_close": 100.0,
        "predicted_close": 101.0,
        "rank": 1,
        "selected": 1,
    }])
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    candidates.to_csv(tmp_path / "candidates.csv", index=False)
    (tmp_path / "history.csv").write_text("", encoding="utf-8")

    monkeypatch.setattr(validation, "PREDICTIONS_FILE", tmp_path / "predictions.csv")
    monkeypatch.setattr(validation, "CANDIDATES_FILE", tmp_path / "candidates.csv")
    monkeypatch.setattr(validation, "HISTORY_FILE", tmp_path / "history.csv")
    monkeypatch.setattr(validation, "OUTPUT_FILE", tmp_path / "validation.csv")
    monkeypatch.setattr(validation, "COVERAGE_FILE", tmp_path / "coverage.json")

    result = validation.run_selection_validation().iloc[0]
    coverage = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))

    assert result["validation_status"] == "blocked_missing_actual_history"
    assert result["promotion_evidence"] == False
    assert coverage["prediction_sessions"] == 13
    assert coverage["candidate_sessions"] == 1
    assert coverage["missing_candidate_session_count"] == 12
    assert coverage["history_rows"] == 0
    assert coverage["matched_candidate_rows"] == 0
    assert coverage["candidate_history_reconstructed_from_top10_predictions"] is False
    assert coverage["missing_facts_inferred"] is False

def test_update_history_recovers_zero_byte_csv_from_source(monkeypatch, tmp_path):
    history_file = tmp_path / "ohlcv.csv"
    history_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(pipeline, "HISTORY_FILE", history_file)

    def fake_download(symbols, start):
        assert start == "2018-01-01"
        return pd.DataFrame(_rows("2026-11-13", symbols))

    monkeypatch.setattr(pipeline, "download_history", fake_download)
    result = pipeline.update_history(["AAA", "BBB"])

    assert len(result) == 2
    assert set(result["symbol"]) == {"AAA", "BBB"}
    assert len(pd.read_csv(history_file)) == 2


def test_update_history_does_not_replace_empty_history_when_source_is_empty(monkeypatch, tmp_path):
    history_file = tmp_path / "ohlcv.csv"
    history_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(pipeline, "HISTORY_FILE", history_file)
    monkeypatch.setattr(
        pipeline, "download_history",
        lambda symbols, start: pd.DataFrame(columns=pipeline.EMPTY_HISTORY),
    )

    import pytest
    with pytest.raises(RuntimeError, match="returned no OHLCV rows"):
        pipeline.update_history(["AAA"])

    assert history_file.read_text(encoding="utf-8") == ""

def _candidate_session_rows(count=12):
    return pd.DataFrame([
        {
            "prediction_date": "2026-11-16",
            "target_date": "2026-11-17",
            "symbol": f"STOCK{i:02d}",
            "rank": i + 1,
            "selected": int(i < 10),
            "base_close": 100.0 + i,
            "predicted_close": 101.0 + i,
            "selection_method": "ranking_top10",
        }
        for i in range(count)
    ])


def test_candidate_session_integrity_accepts_complete_candidate_pool():
    candidates = _candidate_session_rows(12)

    pipeline._validate_candidate_session(candidates, pd.Timestamp("2026-11-17"))


def test_candidate_session_integrity_rejects_incomplete_pool():
    import pytest

    candidates = _candidate_session_rows(9)
    with pytest.raises(RuntimeError, match="at least 10 candidate rows"):
        pipeline._validate_candidate_session(candidates, pd.Timestamp("2026-11-17"))


def test_candidate_session_integrity_rejects_duplicate_symbols_and_invalid_selection_count():
    import pytest

    duplicate = _candidate_session_rows(12)
    duplicate.loc[1, "symbol"] = duplicate.loc[0, "symbol"]
    with pytest.raises(RuntimeError, match="duplicate symbols"):
        pipeline._validate_candidate_session(duplicate, pd.Timestamp("2026-11-17"))

    wrong_selection_count = _candidate_session_rows(12)
    wrong_selection_count.loc[9, "selected"] = 0
    with pytest.raises(RuntimeError, match="exactly 10 explicit selections"):
        pipeline._validate_candidate_session(wrong_selection_count, pd.Timestamp("2026-11-17"))

def test_candidate_history_rerun_replaces_entire_vintage_candidate_pool():
    existing = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "AAA", "rank": 1},
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "BBB", "rank": 2},
        {"prediction_date": "2026-11-12", "target_date": "2026-11-16", "symbol": "CCC", "rank": 1},
    ])
    rerun = pd.DataFrame([
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "AAA", "rank": 1},
        {"prediction_date": "2026-11-13", "target_date": "2026-11-16", "symbol": "DDD", "rank": 2},
    ])

    result = _merge_candidate_history(existing, rerun)

    current = result[result["prediction_date"].astype(str).str.startswith("2026-11-13")]
    assert set(current["symbol"]) == {"AAA", "DDD"}
    assert "BBB" not in set(result["symbol"])
    assert "CCC" in set(result["symbol"])

def test_market_risk_context_accepts_supported_levels_and_bounded_values():
    for level, multiplier in [("RISK_ON", 1.0), ("NEUTRAL", 0.80), ("RISK_OFF", 0.55)]:
        pipeline._validate_market_risk_context(level, multiplier, 50.0)


def test_market_risk_context_rejects_unknown_or_malformed_values():
    import pytest

    with pytest.raises(RuntimeError, match="Unknown market risk level"):
        pipeline._validate_market_risk_context("PANIC", 0.55, 50.0)
    with pytest.raises(RuntimeError, match="Invalid market risk multiplier"):
        pipeline._validate_market_risk_context("RISK_OFF", 1.2, 50.0)
    with pytest.raises(RuntimeError, match="Invalid market intelligence score"):
        pipeline._validate_market_risk_context("NEUTRAL", 0.80, float("nan"))



def test_risk_off_market_selection_fails_closed_when_defensive_selector_falls_back(monkeypatch):
    import pytest

    candidates = pd.DataFrame({"symbol": ["A", "B"], "rank": [1, 2]})
    monkeypatch.setattr(
        pipeline,
        "apply_risk_off_selection",
        lambda frame, risk_level, top_n: (frame, "ranking_top10"),
    )
    with pytest.raises(RuntimeError, match="RISK_OFF market regime"):
        pipeline._apply_market_risk_selection(candidates, "RISK_OFF", 2)


def test_risk_off_market_selection_accepts_defensive_selector(monkeypatch):
    candidates = pd.DataFrame({"symbol": ["A", "B"], "rank": [1, 2]})
    monkeypatch.setattr(
        pipeline,
        "apply_risk_off_selection",
        lambda frame, risk_level, top_n: (frame.iloc[::-1].copy(), "risk_off_defensive_v1"),
    )
    selected, method = pipeline._apply_market_risk_selection(candidates, "RISK_OFF", 2)
    assert method == "risk_off_defensive_v1"
    assert selected["symbol"].tolist() == ["B", "A"]

def test_empty_yahoo_batch_retries_each_symbol_individually(monkeypatch):
    calls = []

    def fake_download(batch, start):
        calls.append(tuple(batch))
        if len(batch) > 1:
            return pd.DataFrame()
        index = pd.DatetimeIndex(["2024-01-02"], name="Date")
        return pd.DataFrame(
            {
                "Open": [10.0], "High": [11.0], "Low": [9.0],
                "Close": [10.5], "Volume": [1000],
                "Dividends": [0.0], "Stock Splits": [0.0],
            },
            index=index,
        )

    monkeypatch.setattr(pipeline, "_download_yahoo_batch", fake_download)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)

    result = pipeline.download_history(["AAA", "BBB"], "2018-01-01")

    assert calls == [("AAA", "BBB"), ("AAA",), ("BBB",)]
    assert set(result["symbol"]) == {"AAA", "BBB"}
    assert len(result) == 2
    assert result["date"].notna().all()


def test_empty_isolated_yahoo_response_is_retried_three_times(monkeypatch):
    calls = []

    def fake_download(batch, start):
        calls.append(tuple(batch))
        return pd.DataFrame()

    monkeypatch.setattr(pipeline, "_download_yahoo_batch", fake_download)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)

    result = pipeline.download_history(["AAA"], "2018-01-01")

    assert calls == [("AAA",), ("AAA",), ("AAA",), ("AAA",)]
    assert result.empty


def test_index_market_features_join_same_date_without_future_lookahead():
    stock = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"]),
        "symbol": ["AAA", "AAA", "AAA"],
    })
    index = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06"]),
        "index": ["NIFTY50"] * 4,
        "close": [100.0, 101.0, 99.99, 102.0],
    })
    out = pipeline.merge_index_market_features(stock, index)
    first = out.iloc[0]
    second = out.iloc[1]
    assert first["index_nifty50_return_1d"] == pytest.approx(0.01)
    assert first["index_nifty50_available_1d"] == 1.0
    assert second["index_nifty50_return_1d"] == pytest.approx(99.99 / 101.0 - 1.0)
    assert second["index_nifty50_return_1d"] != pytest.approx(102.0 / 99.99 - 1.0)
    assert second["index_banknifty_return_1d"] == 0.0
    assert second["index_banknifty_available_1d"] == 0.0


def test_index_market_features_have_explicit_missingness_and_stable_schema():
    stock = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "symbol": ["AAA"]})
    out = pipeline.merge_index_market_features(stock, pd.DataFrame())
    assert set(pipeline.INDEX_MARKET_FEATURE_COLUMNS).issubset(out.columns)
    assert out[pipeline.INDEX_MARKET_FEATURE_COLUMNS].eq(0.0).all().all()
    assert len(pipeline.FEATURE_COLUMNS) == 18 + len(pipeline.INDEX_MARKET_FEATURE_COLUMNS)


def test_index_market_features_ignore_unconfigured_index_symbols():
    stock = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "symbol": ["AAA"]})
    index = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "index": ["UNCONFIGURED", "UNCONFIGURED"],
        "close": [100.0, 110.0],
    })
    out = pipeline.merge_index_market_features(stock, index)
    assert out["index_nifty50_available_1d"].iloc[0] == 0.0
    assert out["index_nifty50_return_1d"].iloc[0] == 0.0


def test_pipeline_features_include_same_day_index_market_context(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=30)
    stock = pd.DataFrame({
        "date": dates, "symbol": ["AAA"] * len(dates),
        "open": [100 + i for i in range(30)], "high": [102 + i for i in range(30)],
        "low": [99 + i for i in range(30)], "close": [101 + i for i in range(30)],
        "volume": [1000] * 30,
    })
    index = pd.DataFrame({
        "date": dates, "index": ["NIFTY50"] * len(dates),
        "close": [20000 + 100 * i for i in range(30)],
    })
    monkeypatch.setattr(pipeline, "INDEX_HISTORY_FILE", tmp_path / "index_ohlcv.csv")
    index.to_csv(pipeline.INDEX_HISTORY_FILE, index=False)
    out = pipeline.features(stock)
    assert set(pipeline.INDEX_MARKET_FEATURE_COLUMNS).issubset(out.columns)
    assert out.iloc[-1]["index_nifty50_available_1d"] == 1.0
    assert out.iloc[-1]["index_nifty50_available_5d"] == 1.0
    assert out.iloc[-1]["index_nifty50_return_1d"] == pytest.approx(100 / 22900)
