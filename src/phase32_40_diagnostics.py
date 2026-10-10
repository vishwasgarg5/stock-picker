from __future__ import annotations

"""Phases 32-40: diagnostics-only evidence, data-quality and monitoring reports.

These reports never alter predictions, filters, position sizing, production ranking,
risk configuration, or champion selection. Historical prediction scoring evaluates
already-recorded predictions; it is not a model-refit walk-forward simulation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_HOLDOUT_DATES = 5
MIN_MATCHED_SESSIONS = 20
MIN_V2_EXECUTED_TRADES = 50


def _read(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()


def _write(frame: pd.DataFrame, name: str) -> None:
    DATA.mkdir(exist_ok=True)
    frame.to_csv(DATA / name, index=False)


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "status": "INSUFFICIENT_DATA"}
    out: dict[str, Any] = {"rows": int(len(frame))}
    for key, col in (("open", "open_ape"), ("high", "high_ape"),
                     ("low", "low_ape"), ("close", "close_ape")):
        if col in frame:
            vals = _safe_num(frame[col]).dropna()
            out[f"{key}_mape_pct"] = float(vals.mean() * 100) if len(vals) else None
    if "close_direction_correct" in frame:
        vals = _safe_num(frame["close_direction_correct"]).dropna()
        out["close_direction_accuracy_pct"] = float(vals.mean() * 100) if len(vals) else None
    out["status"] = "SCORED" if len(frame) >= 20 else "SMALL_SAMPLE"
    return out


def score_historical_predictions(
    candidates: pd.DataFrame,
    actuals: pd.DataFrame,
    min_holdout_dates: int = MIN_HOLDOUT_DATES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score saved predictions against target-date OHLC, split by target date."""
    required = {"prediction_date", "target_date", "symbol", "base_close",
                "predicted_open", "predicted_high", "predicted_low", "predicted_close"}
    actual_required = {"date", "symbol", "open", "high", "low", "close"}
    if candidates.empty or not required.issubset(candidates.columns) or not actual_required.issubset(actuals.columns):
        return pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "scored_rows": 0, "holdout_dates": 0,
            "method": "saved prediction replay; no model refit performed",
        }

    p = candidates.copy()
    a = actuals[list(actual_required)].copy()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce").dt.normalize()
    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
    a["symbol"] = a["symbol"].astype(str).str.upper().str.strip()
    p = p.dropna(subset=["prediction_date", "target_date"])
    order_valid = p["prediction_date"] < p["target_date"]
    invalid_order_rows = int((~order_valid).sum())
    p = p.loc[order_valid].copy()
    p = p.drop_duplicates(["target_date", "symbol"], keep="last")
    a = a.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    a = a.rename(columns={"date": "target_date", "open": "actual_open", "high": "actual_high",
                          "low": "actual_low", "close": "actual_close"})
    scored = p.merge(a, on=["target_date", "symbol"], how="inner", validate="many_to_one")
    if scored.empty:
        return scored, {
            "status": "INSUFFICIENT_DATA", "scored_rows": 0, "holdout_dates": 0,
            "invalid_prediction_order_rows": invalid_order_rows,
            "method": "saved prediction replay; no model refit performed",
        }

    mapping = {
        "open": ("predicted_open", "actual_open"),
        "high": ("predicted_high", "actual_high"),
        "low": ("predicted_low", "actual_low"),
        "close": ("predicted_close", "actual_close"),
    }
    for label, (pred_col, actual_col) in mapping.items():
        scored[pred_col] = _safe_num(scored[pred_col])
        scored[actual_col] = _safe_num(scored[actual_col])
        scored[f"{label}_ae"] = (scored[pred_col] - scored[actual_col]).abs()
        scored[f"{label}_ape"] = scored[f"{label}_ae"] / scored[actual_col].abs().replace(0, np.nan)
    scored["base_close"] = _safe_num(scored["base_close"])
    scored["baseline_close_ape"] = (
        (scored["base_close"] - scored["actual_close"]).abs()
        / scored["actual_close"].abs().replace(0, np.nan)
    )
    scored["close_direction_correct"] = (
        (scored["predicted_close"] >= scored["base_close"])
        == (scored["actual_close"] >= scored["base_close"])
    ).astype(int)

    dates = sorted(scored["target_date"].dropna().unique())
    if len(dates) >= 2:
        holdout_idx = min(len(dates) - 1, max(1, int(np.ceil(len(dates) * 0.60))))
        holdout_start = pd.Timestamp(dates[holdout_idx])
        scored["period"] = np.where(scored["target_date"] < holdout_start, "EARLY_CONTEXT", "CHRONOLOGICAL_HOLDOUT")
    else:
        holdout_start = None
        scored["period"] = "INSUFFICIENT_DATES"

    context = scored[scored["period"] == "EARLY_CONTEXT"]
    holdout = scored[scored["period"] == "CHRONOLOGICAL_HOLDOUT"]
    holdout_date_count = int(holdout["target_date"].nunique())
    summary = {
        "status": "SCORED" if len(holdout) >= 20 and holdout_date_count >= min_holdout_dates and holdout_start is not None else "INSUFFICIENT_HOLDOUT",
        "scored_rows": int(len(scored)),
        "unique_target_dates": int(scored["target_date"].nunique()),
        "holdout_target_dates": holdout_date_count,
        "minimum_holdout_target_dates_required": int(min_holdout_dates),
        "invalid_prediction_order_rows": invalid_order_rows,
        "first_target_date": scored["target_date"].min().date().isoformat(),
        "last_target_date": scored["target_date"].max().date().isoformat(),
        "holdout_start_date": holdout_start.date().isoformat() if holdout_start is not None else None,
        "context_metrics": _metrics(context),
        "holdout_metrics": _metrics(holdout),
        "holdout_baseline_close_mape_pct": float(holdout["baseline_close_ape"].mean() * 100) if not holdout.empty else None,
        "holdout_prediction_close_mape_pct": float(holdout["close_ape"].mean() * 100) if not holdout.empty else None,
        "method": "chronological replay of already-recorded predictions; no model refit performed",
        "promotion_allowed": False,
    }
    return scored, summary


def _phase32_funnel(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = DATA / "phase32_v2_selection_funnel.csv"
    if path.exists():
        funnel = _read("phase32_v2_selection_funnel.csv")
        if not funnel.empty and {"stage", "input_rows", "output_rows"}.issubset(funnel.columns):
            summary = {"status": "OBSERVED", "stages": int(len(funnel)),
                       "final_selected_rows": int(pd.to_numeric(funnel.iloc[-1]["output_rows"], errors="coerce") or 0)}
            return funnel, summary
    if not trades.empty and "signal" in trades:
        signals = trades["signal"].astype(str).str.upper().value_counts()
        skips = trades[trades["signal"].astype(str).str.upper().eq("SKIP")]
        reason_counts = skips.get("no_trade_reason", pd.Series(dtype=str)).fillna("unspecified").value_counts()
        rows = [{"stage": "existing_trade_history", "input_rows": int(len(trades)),
                 "output_rows": int(signals.get("BUY", 0)), "rejected_rows": int(signals.get("SKIP", 0)),
                 "detail": "Fallback only: existing trade history cannot identify every internal filter stage"}]
        rows.extend({"stage": f"skip_reason:{reason}", "input_rows": int(n), "output_rows": 0,
                     "rejected_rows": int(n), "detail": "Recorded no-trade reason"} for reason, n in reason_counts.items())
        return pd.DataFrame(rows), {"status": "FALLBACK_HISTORY_ONLY", "buy_rows": int(signals.get("BUY", 0)),
                                    "skip_rows": int(signals.get("SKIP", 0))}
    return pd.DataFrame([{"stage": "selection_funnel", "input_rows": 0, "output_rows": 0,
                          "rejected_rows": 0, "detail": "Awaiting next V2 run; no funnel data available"}]), {
        "status": "AWAITING_NEXT_V2_RUN"
    }


def _phase35_trade_metrics(trades: pd.DataFrame, model: str) -> dict[str, Any]:
    if trades.empty or "signal" not in trades:
        return {"model": model, "executed_trades": 0, "status": "NO_DATA"}
    x = trades[trades["signal"].astype(str).str.upper().eq("BUY")].copy()
    qty_col = "quantity" if "quantity" in x else ("shares" if "shares" in x else None)
    if qty_col:
        x = x[_safe_num(x[qty_col]).fillna(0) > 0]
    pnl_col = "profit_loss" if "profit_loss" in x else None
    if pnl_col:
        pnl = _safe_num(x[pnl_col]).fillna(0)
    else:
        pnl = pd.Series(0.0, index=x.index)
    cost_col = "trading_cost" if "trading_cost" in x else ("costs" if "costs" in x else None)
    costs = _safe_num(x[cost_col]).fillna(0) if cost_col else pd.Series(0.0, index=x.index)
    gross_col = "gross_profit_loss" if "gross_profit_loss" in x else None
    gross = _safe_num(x[gross_col]).fillna(0) if gross_col else pnl + costs
    ordered = x.copy()
    ordered["_net_pnl"] = pnl
    if "target_date" in ordered:
        ordered["_date"] = pd.to_datetime(ordered["target_date"], errors="coerce")
        ordered = ordered.sort_values("_date")
    ordered["_portfolio_value"] = 100000.0 + ordered["_net_pnl"].cumsum()
    peak = ordered["_portfolio_value"].cummax()
    drawdown_pct = (ordered["_portfolio_value"] / peak - 1.0) * 100 if len(ordered) else pd.Series(dtype=float)
    net_total = float(pnl.sum())
    cost_total = float(costs.sum())
    out: dict[str, Any] = {
        "model": model, "executed_trades": int(len(x)), "gross_pnl": float(gross.sum()),
        "net_pnl": net_total, "recorded_costs": cost_total,
        "net_pnl_with_50pct_extra_cost_stress": net_total - 0.5 * cost_total,
        "net_pnl_with_100pct_extra_cost_stress": net_total - cost_total,
        "max_trade_sequence_drawdown_pct": float(drawdown_pct.min()) if len(drawdown_pct) else None,
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(x) else None,
        "expectancy_per_trade": float(pnl.mean()) if len(x) else None,
        "status": "SMALL_SAMPLE" if len(x) < MIN_V2_EXECUTED_TRADES else "SAMPLE_GATE_MET",
        "costs_note": "Uses recorded costs and net profit_loss; stress cases subtract only additional assumed costs",
    }
    if "return_pct" in x:
        returns = _safe_num(x["return_pct"]).dropna()
        out["mean_trade_return_pct"] = float(returns.mean()) if len(returns) else None
    return out


def run() -> dict[str, Any]:
    DATA.mkdir(exist_ok=True)
    candidates = _read("prediction_candidates_history.csv")
    if candidates.empty:
        candidates = _read("predictions.csv")
    actuals = _read("ohlcv.csv", ["date", "symbol", "open", "high", "low", "close"])
    v1_trades = _read("paper_trades.csv")
    v2_trades = _read("paper_trades_v2.csv")
    v1_daily = _read("portfolio_daily.csv")
    v2_daily = _read("portfolio_v2_daily.csv")
    index_news = _read("index_intelligence_summary.csv")

    # Phase 32: selection funnel. Exact gate counts appear after an instrumented V2 run.
    funnel, funnel_summary = _phase32_funnel(v2_trades)
    _write(funnel, "phase32_selection_funnel_report.csv")

    # Phase 33: leakage-checked chronological scoring of historical predictions.
    scored, prediction_summary = score_historical_predictions(candidates, actuals)
    _write(scored, "phase33_chronological_prediction_scores.csv")
    (DATA / "phase33_prediction_summary.json").write_text(json.dumps(prediction_summary, indent=2, default=str), encoding="utf-8")

    # Phase 34: descriptive accuracy by an available, non-empty risk/regime bucket.
    regime_report = pd.DataFrame()
    if not scored.empty:
        bucket_col = next((col for col in ("regime", "risk_level", "confidence_tier", "volatility20")
                           if col in scored.columns and scored[col].notna().any()), None)
        if bucket_col:
            records = []
            for bucket, group in scored.groupby(bucket_col, dropna=False):
                row = {"bucket_field": bucket_col, "bucket": str(bucket), "period": "ALL"}
                row.update(_metrics(group))
                records.append(row)
            regime_report = pd.DataFrame(records)
        else:
            spread = _safe_num(scored.get("prediction_spread", pd.Series(np.nan, index=scored.index)))
            if spread.notna().any():
                scored["spread_bucket"] = pd.cut(spread, [-np.inf, 0.005, 0.015, np.inf],
                                                  labels=["LOW_SPREAD", "MID_SPREAD", "HIGH_SPREAD"])
                records = []
                for bucket, group in scored.groupby("spread_bucket", observed=True):
                    row = {"bucket_field": "prediction_spread", "bucket": str(bucket), "period": "ALL"}
                    row.update(_metrics(group))
                    records.append(row)
                regime_report = pd.DataFrame(records)
    if regime_report.empty:
        regime_report = pd.DataFrame([{"bucket_field": "none", "bucket": "unavailable",
                                       "status": "INSUFFICIENT_DATA", "rows": 0}])
    _write(regime_report, "phase34_regime_accuracy.csv")

    # Phase 35: recorded net trade results and costs; never subtract recorded costs twice.
    trade_report = pd.DataFrame([
        _phase35_trade_metrics(v1_trades, "V1"),
        _phase35_trade_metrics(v2_trades, "V2"),
    ])
    _write(trade_report, "phase35_trading_costs_and_returns.csv")

    # Phase 36: evidence gate only; it cannot promote a model.
    phase31_path = DATA / "phase31_summary.json"
    try:
        phase31 = json.loads(phase31_path.read_text(encoding="utf-8")) if phase31_path.exists() else {}
    except (OSError, ValueError):
        phase31 = {}
    v2_metric = next((x for x in trade_report.to_dict("records") if x["model"] == "V2"), {})
    matched = int(phase31.get("matched_sessions", 0) or 0)
    v2_executed = int(v2_metric.get("executed_trades", 0) or 0)
    v1_holdout = phase31.get("holdout_v1_net_pnl")
    v2_holdout = phase31.get("holdout_v2_net_pnl")
    comparable = v1_holdout is not None and v2_holdout is not None
    evidence_ready = (matched >= MIN_MATCHED_SESSIONS and v2_executed >= MIN_V2_EXECUTED_TRADES
                      and comparable and float(v2_holdout) > 0 and float(v2_holdout) > float(v1_holdout))
    promotion = {
        "matched_sessions": matched, "minimum_matched_sessions": MIN_MATCHED_SESSIONS,
        "v2_executed_trades": v2_executed, "minimum_v2_executed_trades": MIN_V2_EXECUTED_TRADES,
        "holdout_comparison_available": comparable,
        "holdout_v1_net_pnl": v1_holdout, "holdout_v2_net_pnl": v2_holdout,
        "status": "ELIGIBLE_FOR_HUMAN_REVIEW" if evidence_ready else "NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE",
        "production_champion": "V1", "v2_promoted": False,
        "automatic_promotion": False,
    }
    (DATA / "phase36_promotion_gate.json").write_text(json.dumps(promotion, indent=2, default=str), encoding="utf-8")

    # Phase 37: input integrity and freshness checks.
    data_checks: list[dict[str, Any]] = []
    if actuals.empty:
        data_checks.append({"check": "ohlcv_available", "status": "BLOCK", "affected_rows": 0, "detail": "OHLCV missing or unreadable"})
    else:
        data_checks.append({"check": "ohlcv_available", "status": "PASS", "affected_rows": 0, "detail": f"{len(actuals)} OHLCV rows read"})
        dates = pd.to_datetime(actuals["date"], errors="coerce")
        symbols = actuals["symbol"].astype(str).str.upper().str.strip()
        valid_date = dates.notna()
        data_checks.append({"check": "ohlcv_valid_dates", "status": "PASS" if valid_date.all() else "WARN",
                            "affected_rows": int((~valid_date).sum()), "detail": "date must parse"})
        duplicate = pd.DataFrame({"date": dates.dt.normalize(), "symbol": symbols}).duplicated(["date", "symbol"], keep=False)
        data_checks.append({"check": "ohlcv_unique_symbol_date", "status": "PASS" if not duplicate.any() else "WARN",
                            "affected_rows": int(duplicate.sum()), "detail": "duplicate date/symbol rows"})
        price_cols = ["open", "high", "low", "close"]
        numeric = actuals[price_cols].apply(pd.to_numeric, errors="coerce")
        bad_price = numeric.isna().any(axis=1) | (numeric <= 0).any(axis=1)
        bad_ohlc = (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)) | (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1))
        data_checks.append({"check": "ohlcv_positive_prices", "status": "PASS" if not bad_price.any() else "WARN",
                            "affected_rows": int(bad_price.sum()), "detail": "OHLC prices must be numeric and positive"})
        data_checks.append({"check": "ohlc_consistency", "status": "PASS" if not bad_ohlc.any() else "WARN",
                            "affected_rows": int(bad_ohlc.sum()), "detail": "high/low must bound open, close and range"})
        latest = dates.max()
        age_days = int((pd.Timestamp.now().normalize() - latest.normalize()).days) if pd.notna(latest) else None
        data_checks.append({"check": "ohlcv_freshness", "status": "PASS" if age_days is not None and age_days <= 5 else "REVIEW",
                            "affected_rows": 0, "detail": f"latest date={latest}; calendar age days={age_days}; weekends/holidays not adjusted"})
    _write(pd.DataFrame(data_checks), "phase37_data_reliability.csv")

    # Phase 38: source and timestamp hygiene; sentiment is not claimed predictive from this audit.
    if index_news.empty:
        news_report = pd.DataFrame([{"check": "news_data_available", "status": "REVIEW", "affected_rows": 0,
                                     "detail": "No index-intelligence summary available"}])
    else:
        headlines = index_news.get("news_headline", pd.Series("", index=index_news.index)).fillna("").astype(str).str.strip()
        urls = index_news.get("news_source_url", pd.Series("", index=index_news.index)).fillna("").astype(str).str.strip()
        as_of = pd.to_datetime(index_news.get("as_of", pd.Series(pd.NaT, index=index_news.index)), errors="coerce")
        news_report = pd.DataFrame([
            {"check": "news_rows", "status": "PASS", "affected_rows": 0, "detail": f"{len(index_news)} index-intelligence rows"},
            {"check": "headline_present", "status": "PASS" if headlines.ne("").all() else "WARN",
             "affected_rows": int(headlines.eq("").sum()), "detail": "headline field is populated"},
            {"check": "source_url_present", "status": "PASS" if urls.str.startswith(("http://", "https://")).all() else "WARN",
             "affected_rows": int((~urls.str.startswith(("http://", "https://"))).sum()), "detail": "source URLs must be HTTP(S)"},
            {"check": "news_asof_date_parseable", "status": "PASS" if as_of.notna().all() else "WARN",
             "affected_rows": int(as_of.isna().sum()), "detail": "as_of timestamp/date parses"},
            {"check": "duplicate_headlines", "status": "WARN" if headlines[headlines.ne("")].duplicated().any() else "PASS",
             "affected_rows": int(headlines[headlines.ne("")].duplicated(keep=False).sum()), "detail": "exact duplicate headline text; not proof of duplicate event"},
        ])
        # Exploratory next-session sentiment check; no predictive claim unless the
        # source history has enough distinct dates per index to evaluate out of sample.
        alignment_rows = 0
        alignment_correct = 0
        if {"index", "as_of", "change_1d_pct", "news_sentiment"}.issubset(index_news.columns):
            lagged = index_news.copy()
            lagged["as_of"] = pd.to_datetime(lagged["as_of"], errors="coerce").dt.normalize()
            lagged["change_1d_pct"] = _safe_num(lagged["change_1d_pct"])
            sentiment = lagged["news_sentiment"].astype(str).str.upper()
            lagged["_sentiment_sign"] = np.select(
                [sentiment.str.contains("BULL"), sentiment.str.contains("BEAR")],
                [1, -1], default=0
            )
            lagged = lagged.dropna(subset=["as_of", "change_1d_pct"]).sort_values(["index", "as_of"])
            lagged["next_session_change_pct"] = lagged.groupby("index")["change_1d_pct"].shift(-1)
            comparable_news = lagged.dropna(subset=["next_session_change_pct"])
            alignment_rows = int(len(comparable_news))
            if alignment_rows:
                alignment_correct = int((
                    (comparable_news["_sentiment_sign"] > 0) & (comparable_news["next_session_change_pct"] > 0)
                    | (comparable_news["_sentiment_sign"] < 0) & (comparable_news["next_session_change_pct"] < 0)
                ).sum())
        news_report = pd.concat([news_report, pd.DataFrame([{
            "check": "exploratory_next_session_sentiment_alignment",
            "status": "INSUFFICIENT_DATA" if alignment_rows < 20 else "DESCRIPTIVE_ONLY",
            "affected_rows": alignment_rows,
            "detail": (f"{alignment_correct}/{alignment_rows} directional matches; not used for model decisions"
                       if alignment_rows else "Need at least 20 as-of index/date pairs; source timestamp quality must be verified")
        }])], ignore_index=True)
    _write(news_report, "phase38_news_source_audit.csv")

    # Phase 39: date-ordered rolling error/drift report.
    drift = pd.DataFrame()
    if not scored.empty:
        daily = scored.groupby("target_date", as_index=False).agg(
            rows=("symbol", "count"), close_mape=("close_ape", "mean"),
            direction_accuracy=("close_direction_correct", "mean"),
            baseline_close_mape=("baseline_close_ape", "mean"),
        ).sort_values("target_date")
        daily["rolling_5_session_close_mape_pct"] = daily["close_mape"].rolling(5, min_periods=3).mean() * 100
        daily["rolling_5_session_direction_accuracy_pct"] = daily["direction_accuracy"].rolling(5, min_periods=3).mean() * 100
        daily["close_mape_pct"] = daily["close_mape"] * 100
        daily["baseline_close_mape_pct"] = daily["baseline_close_mape"] * 100
        daily["drift_flag"] = False
        if len(daily) >= 8:
            early = daily["close_mape"].iloc[:max(3, len(daily) // 2)].mean()
            recent = daily["close_mape"].iloc[-max(3, len(daily) // 2):].mean()
            daily["drift_flag"] = bool(pd.notna(early) and early > 0 and recent > early * 1.25)
        drift = daily.drop(columns=["close_mape", "direction_accuracy", "baseline_close_mape"])
    if drift.empty:
        drift = pd.DataFrame([{"status": "INSUFFICIENT_DATA", "drift_flag": False,
                               "detail": "Need scored historical predictions for rolling drift report"}])
    _write(drift, "phase39_prediction_drift.csv")

    # Phase 40: one concise status object joining all diagnostic phases.
    status_counts = {}
    for frame in (pd.DataFrame(data_checks), news_report, regime_report):
        if "status" in frame:
            status_counts.update({str(k): int(v) for k, v in frame["status"].value_counts().items()})
    dashboard = {
        "phases": "32-40", "diagnostics_only": True,
        "phase32_selection_funnel": funnel_summary,
        "phase33_chronological_prediction_replay": prediction_summary,
        "phase34_regime_accuracy_rows": int(len(regime_report)),
        "phase35_trade_metrics": trade_report.to_dict("records"),
        "phase36_promotion_gate": promotion,
        "phase37_data_checks": data_checks,
        "phase38_news_checks": news_report.to_dict("records"),
        "phase39_drift_rows": int(len(drift)),
        "production_champion": "V1", "v2_promoted": False,
        "production_model_changed": False, "ranking_or_risk_changed": False,
        "status": "REVIEW_REQUIRED" if (
            prediction_summary.get("status") != "SCORED"
            or promotion.get("status") != "ELIGIBLE_FOR_HUMAN_REVIEW"
            or any(x.get("status") in {"BLOCK", "WARN", "REVIEW", "INSUFFICIENT_DATA"} for x in data_checks + news_report.to_dict("records"))
        ) else "DIAGNOSTICS_GENERATED",
        "status_counts": status_counts,
    }
    (DATA / "phase40_dashboard.json").write_text(json.dumps(dashboard, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Stock Picker Phases 32-40 — Diagnostic Dashboard", "",
        f"- Overall status: **{dashboard['status']}**",
        "- Production champion: **V1**",
        "- V2 promotion: **No**",
        "- Production model/ranking/risk changed: **No**", "",
        "## Selection funnel", f"- Status: {funnel_summary.get('status')}",
        f"- Details: {funnel_summary}", "",
        "## Historical prediction replay", f"- Status: {prediction_summary.get('status')}",
        f"- Scored rows: {prediction_summary.get('scored_rows', 0)}",
        f"- Holdout metrics: {prediction_summary.get('holdout_metrics')}", "",
        "## Trading evidence", f"- Promotion gate: {promotion['status']}",
        f"- Matched sessions: {matched}; V2 executed trades: {v2_executed}", "",
        "## Data and news audits", f"- Data checks: {len(data_checks)}",
        f"- News checks: {len(news_report)}", "",
        "Note: saved-prediction replay is not a model-refit walk-forward backtest. These reports do not change strategy decisions.",
    ]
    (DATA / "phase40_dashboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": dashboard["status"], "phase32": funnel_summary,
                      "phase33": prediction_summary.get("status"),
                      "phase36": promotion["status"], "production_champion": "V1",
                      "v2_promoted": False}, indent=2, default=str))
    return dashboard


if __name__ == "__main__":
    run()
