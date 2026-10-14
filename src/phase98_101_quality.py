from __future__ import annotations

"""Phases 98-101: forecast diagnosis, point-in-time features, shadow challenger,
confidence/selection outcome audit, and paper-trading performance. Analysis only.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_TRAIN_ROWS = 250
FEATURES = ["ret_1", "ret_5", "ret_20", "volatility_20", "volume_ratio_20"]


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(np.nan, index=frame.index)


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    a, p = pd.to_numeric(actual, errors="coerce"), pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna() & a.ne(0)
    return float((p[valid] - a[valid]).abs().div(a[valid].abs()).mean() * 100) if valid.any() else None


def _feature_panel(history: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close", "volume"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    h = history[list(required)].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    for col in ("close", "volume"):
        h[col] = pd.to_numeric(h[col], errors="coerce")
    h = h.dropna(subset=["date", "symbol", "close"]).query("close > 0")
    h = h.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    groups = h.groupby("symbol", sort=False)
    h["ret_1"] = groups["close"].pct_change(1)
    h["ret_5"] = groups["close"].pct_change(5)
    h["ret_20"] = groups["close"].pct_change(20)
    h["volatility_20"] = groups["close"].transform(lambda s: s.pct_change().rolling(20, min_periods=15).std())
    h["volume_mean_20"] = groups["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    h["volume_ratio_20"] = h["volume"] / h["volume_mean_20"].replace(0, np.nan)
    # The label belongs to the next observed session. Features remain on date t.
    h["target_date"] = groups["date"].shift(-1)
    h["target_return"] = groups["close"].shift(-1) / h["close"] - 1
    return h


def _diagnose_forecasts(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"}
    if scores.empty or not required.issubset(scores.columns):
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "scored_rows": 0}
    x = scores.copy()
    x["prediction_date"] = pd.to_datetime(x["prediction_date"], errors="coerce").dt.normalize()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    for c in ("base_close", "predicted_close", "actual_close", "predicted_open", "predicted_high", "predicted_low", "actual_open", "actual_high", "actual_low"):
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["prediction_date", "target_date", "symbol", "base_close", "predicted_close", "actual_close"])
    x = x[x["target_date"] > x["prediction_date"]].copy()
    x["actual_return_pct"] = (x["actual_close"] / x["base_close"] - 1) * 100
    x["model_error_pct"] = (x["predicted_close"] / x["actual_close"] - 1) * 100
    x["absolute_error_pct"] = x["model_error_pct"].abs()
    x["actual_move_abs_pct"] = x["actual_return_pct"].abs()
    x["volatility_bucket"] = pd.cut(x["actual_move_abs_pct"], [-np.inf, 1, 2, 4, np.inf], labels=["LOW_MOVE_<=1%", "1_TO_2%", "2_TO_4%", "HIGH_MOVE_>4%"])
    x["bias"] = np.select([x["model_error_pct"] > 0, x["model_error_pct"] < 0], ["OVERPREDICT", "UNDERPREDICT"], default="EXACT")
    group_cols = ["symbol"]
    by_symbol = x.groupby(group_cols, observed=True).agg(
        rows=("absolute_error_pct", "size"),
        close_mape_pct=("absolute_error_pct", "mean"),
        signed_error_pct=("model_error_pct", "mean"),
        mean_actual_move_abs_pct=("actual_move_abs_pct", "mean"),
        overprediction_rate_pct=("bias", lambda s: float((s == "OVERPREDICT").mean() * 100)),
    ).reset_index().sort_values(["close_mape_pct", "rows"], ascending=[False, False])
    segment_cols = ["volatility_bucket"]
    if "period" in x:
        x["period"] = x["period"].fillna("UNKNOWN").astype(str)
        segment_cols.append("period")
    by_segment = x.groupby(segment_cols, observed=True, dropna=False).agg(
        rows=("absolute_error_pct", "size"),
        close_mape_pct=("absolute_error_pct", "mean"),
        signed_error_pct=("model_error_pct", "mean"),
        overprediction_rate_pct=("bias", lambda s: float((s == "OVERPREDICT").mean() * 100)),
    ).reset_index()
    ohlc_metrics = {}
    for label in ("open", "high", "low", "close"):
        pred_col, actual_col = f"predicted_{label}", f"actual_{label}"
        if pred_col in x and actual_col in x:
            ohlc_metrics[f"{label}_mape_pct"] = _mape(x[actual_col], x[pred_col])
    summary = {
        "status": "SCORED" if len(x) >= 20 else ("SMALL_SAMPLE" if len(x) else "INSUFFICIENT_DATA"),
        "scored_rows": int(len(x)), "symbols": int(x["symbol"].nunique()),
        "ohlc_mape_pct": ohlc_metrics,
        "close_mape_pct": _mape(x["actual_close"], x["predicted_close"]),
        "mean_signed_error_pct": float(x["model_error_pct"].mean()) if len(x) else None,
        "overprediction_rows": int((x["bias"] == "OVERPREDICT").sum()),
        "underprediction_rows": int((x["bias"] == "UNDERPREDICT").sum()),
        "diagnostic_only": True,
    }
    return by_symbol, by_segment, summary


def _walk_forward_candidate(scores: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expanding-window pooled Ridge return model; training labels end before prediction date."""
    required = {"prediction_date", "target_date", "symbol", "base_close", "actual_close"}
    if scores.empty or not required.issubset(scores.columns) or panel.empty:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "candidate_rows": 0, "model_refit_production": False}
    p = scores.copy()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"], errors="coerce").dt.normalize()
    p["target_date"] = pd.to_datetime(p["target_date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
    p["base_close"] = pd.to_numeric(p["base_close"], errors="coerce")
    p["actual_close"] = pd.to_numeric(p["actual_close"], errors="coerce")
    p = p.dropna(subset=["prediction_date", "target_date", "symbol", "base_close", "actual_close"])
    p = p[p["target_date"] > p["prediction_date"]].copy()
    # Use only feature rows that existed at prediction time.
    features = panel.dropna(subset=FEATURES + ["target_date", "target_return"]).copy()
    rows = []
    for pred_date, batch in p.groupby("prediction_date", sort=True):
        train = features.loc[features["target_date"] < pred_date]
        if len(train) < MIN_TRAIN_ROWS or train["target_date"].nunique() < 20:
            continue
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["target_return"])
        if len(train) < MIN_TRAIN_ROWS:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[FEATURES], train["target_return"].clip(-0.25, 0.25))
        # Features at the most recent available date on/before prediction_date.
        feature_at_date = panel.loc[panel["date"].le(pred_date)].sort_values("date").drop_duplicates("symbol", keep="last")
        feature_at_date = feature_at_date.set_index("symbol")
        for _, r in batch.iterrows():
            if r["symbol"] not in feature_at_date.index or not np.isfinite(r["base_close"]) or r["base_close"] <= 0:
                continue
            f = feature_at_date.loc[r["symbol"], FEATURES]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[-1]
            f = pd.to_numeric(f, errors="coerce")
            if f.isna().any() or not np.isfinite(f.to_numpy(dtype=float)).all():
                continue
            ret = float(np.clip(model.predict(pd.DataFrame([f.to_dict()])[FEATURES])[0], -0.25, 0.25))
            rows.append({
                "prediction_date": pred_date, "target_date": r["target_date"], "symbol": r["symbol"],
                "base_close": float(r["base_close"]), "actual_close": float(r["actual_close"]),
                "candidate_close": float(r["base_close"] * (1 + ret)), "candidate_return_pct": ret * 100,
                "training_rows": int(len(train)), "training_latest_target_date": str(train["target_date"].max().date()),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {
            "status": "INSUFFICIENT_HISTORY", "candidate_rows": 0, "minimum_training_rows": MIN_TRAIN_ROWS,
            "model_refit_production": False, "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        }
    summary = {
        "status": "SCORED" if len(out) >= 20 else "SMALL_SAMPLE", "candidate_rows": int(len(out)),
        "candidate_symbols": int(out["symbol"].nunique()), "target_dates": int(out["target_date"].nunique()),
        "candidate_close_mape_pct": _mape(out["actual_close"], out["candidate_close"]),
        "previous_close_baseline_mape_pct": _mape(out["actual_close"], out["base_close"]),
        "candidate_beats_previous_close": bool(_mape(out["actual_close"], out["candidate_close"]) < _mape(out["actual_close"], out["base_close"])),
        "walk_forward": True, "expanding_window": True, "minimum_training_rows": MIN_TRAIN_ROWS,
        "training_labels_strictly_before_prediction_date": True, "model_refit_production": False,
        "ranking_or_risk_changed": False, "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        "warning": "Candidate is exploratory and must beat baselines across repeated holdouts before promotion consideration.",
    }
    return out, summary


def _paper_performance(trades: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trades.empty:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "trade_rows": 0, "source": name}
    x = trades.copy()
    if "signal" in x:
        x = x[x["signal"].astype(str).str.upper().isin(["BUY", "LONG", "EXECUTED"])]
    pnl_col = next((c for c in ("profit_loss", "net_pnl", "daily_profit_loss") if c in x), None)
    if pnl_col is None:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "trade_rows": int(len(x)), "reason": "No recognized net P&L field", "source": name}
    x["net_pnl"] = pd.to_numeric(x[pnl_col], errors="coerce")
    date_col = next((c for c in ("target_date", "date") if c in x), None)
    if date_col:
        x["target_date"] = pd.to_datetime(x[date_col], errors="coerce").dt.normalize()
    else:
        x["target_date"] = pd.NaT
    x = x.dropna(subset=["net_pnl"])
    if x.empty:
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "trade_rows": 0, "source": name}
    daily = x.groupby("target_date", dropna=False, as_index=False).agg(net_pnl=("net_pnl", "sum"), trades=("net_pnl", "size"))
    daily = daily.sort_values("target_date", na_position="last")
    daily["cumulative_pnl"] = daily["net_pnl"].cumsum()
    equity = 100000.0 + daily["cumulative_pnl"]
    daily["drawdown_pct"] = (equity / equity.cummax() - 1) * 100
    wins = x.loc[x["net_pnl"] > 0, "net_pnl"]
    losses = x.loc[x["net_pnl"] < 0, "net_pnl"]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    costs_col = next((c for c in ("costs", "trading_cost") if c in x), None)
    costs = float(pd.to_numeric(x[costs_col], errors="coerce").fillna(0).sum()) if costs_col else None
    summary = {
        "status": "SCORED" if len(x) >= 20 else "SMALL_SAMPLE", "source": name, "executed_trade_rows": int(len(x)),
        "sessions": int(x["target_date"].nunique()), "net_pnl_sum": float(x["net_pnl"].sum()),
        "win_rate_pct": float((x["net_pnl"] > 0).mean() * 100),
        "average_win": float(wins.mean()) if len(wins) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else (None if gross_profit == 0 else "NO_LOSSES_IN_SAMPLE"),
        "reported_costs_sum_not_subtracted_again": costs,
        "max_drawdown_pct_on_100k_reference_equity": float(daily["drawdown_pct"].min()),
        "performance_is_paper_only": True,
    }
    return daily, summary


def _selection_outcomes(v2: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    needed = {"target_date", "symbol", "signal", "reference_price"}
    actual_needed = {"date", "symbol", "close"}
    if v2.empty or not needed.issubset(v2.columns) or history.empty or not actual_needed.issubset(history.columns):
        return pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "rows": 0}
    x = v2.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["reference_price"] = pd.to_numeric(x["reference_price"], errors="coerce")
    h = history[["date", "symbol", "close"]].copy()
    h["target_date"] = pd.to_datetime(h.pop("date"), errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    h["actual_close"] = pd.to_numeric(h.pop("close"), errors="coerce")
    h = h.dropna(subset=["target_date", "symbol"]).drop_duplicates(["target_date", "symbol"], keep="last")
    x = x.merge(h, on=["target_date", "symbol"], how="left", validate="many_to_one")
    x["realized_close_return_pct"] = (x["actual_close"] / x["reference_price"] - 1) * 100
    x["signal_group"] = x["signal"].astype(str).str.upper().map(lambda s: "SELECTED_BUY" if s == "BUY" else "REJECTED_OR_SKIPPED")
    x["positive_close_outcome"] = x["realized_close_return_pct"] > 0
    grouped = x.dropna(subset=["realized_close_return_pct"]).groupby("signal_group", as_index=False).agg(
        rows=("realized_close_return_pct", "size"),
        mean_close_return_pct=("realized_close_return_pct", "mean"),
        median_close_return_pct=("realized_close_return_pct", "median"),
        positive_close_rate_pct=("positive_close_outcome", lambda s: float(s.mean() * 100)),
    )
    confidence_col = next((c for c in ("confidence_score", "confidence_v3", "confidence_pct") if c in x), None)
    confidence_rows = pd.DataFrame()
    if confidence_col:
        x["confidence_value"] = pd.to_numeric(x[confidence_col], errors="coerce")
        valid_conf = x.dropna(subset=["confidence_value", "realized_close_return_pct"]).copy()
        if not valid_conf.empty:
            # Rank-based quantiles tolerate different confidence-score scales and ties.
            ranks = valid_conf["confidence_value"].rank(method="average", pct=True)
            valid_conf["confidence_bucket"] = pd.cut(ranks, [0, .2, .4, .6, .8, 1.000001], labels=["Q1_LOW", "Q2", "Q3", "Q4", "Q5_HIGH"], include_lowest=True)
            confidence_rows = valid_conf.groupby("confidence_bucket", observed=True, as_index=False).agg(
                rows=("realized_close_return_pct", "size"),
                mean_close_return_pct=("realized_close_return_pct", "mean"),
                median_close_return_pct=("realized_close_return_pct", "median"),
                positive_close_rate_pct=("positive_close_outcome", lambda s: float(s.mean() * 100)),
                mean_confidence_score=("confidence_value", "mean"),
            )
            confidence_rows.insert(0, "report_type", "CONFIDENCE_OUTCOME_BUCKET")
    if not grouped.empty:
        grouped.insert(0, "report_type", "SELECTION_OUTCOME")
    combined = pd.concat([grouped, confidence_rows], ignore_index=True, sort=False)
    summary = {
        "status": "SCORED" if len(x.dropna(subset=["realized_close_return_pct"])) >= 20 else "SMALL_SAMPLE",
        "rows": int(len(x)), "rows_with_actuals": int(x["realized_close_return_pct"].notna().sum()),
        "buy_rows": int((x["signal_group"] == "SELECTED_BUY").sum()),
        "skipped_rows": int((x["signal_group"] == "REJECTED_OR_SKIPPED").sum()),
        "confidence_field_used": confidence_col,
        "confidence_buckets_are_rank_quantiles_not_calibrated_probabilities": True,
        "interpretation": "Skipped-name returns are diagnostic counterfactuals, not executable portfolio returns. Confidence buckets are descriptive, not calibrated probabilities.",
        "selection_gates_changed": False,
    }
    return combined, summary



def _v1_loss_attribution(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attribute existing V1 paper net P&L by symbol and exit reason; analysis only."""
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "trade_rows": 0}
    x = trades.copy()
    if "signal" in x:
        x = x[x["signal"].astype(str).str.upper().isin(["BUY", "LONG", "EXECUTED"])].copy()
    pnl_col = next((c for c in ("profit_loss", "net_pnl", "daily_profit_loss") if c in x), None)
    if pnl_col is None or "symbol" not in x:
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "trade_rows": int(len(x)), "reason": "Missing symbol or net P&L field"}
    x["net_pnl"] = pd.to_numeric(x[pnl_col], errors="coerce")
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x = x.dropna(subset=["net_pnl"])
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "trade_rows": 0}
    reason_col = next((c for c in ("exit_reason", "exit_type", "reason") if c in x), None)
    x["exit_reason_group"] = x[reason_col].fillna("UNKNOWN").astype(str).str.strip() if reason_col else "NOT_RECORDED"
    x.loc[x["exit_reason_group"].eq(""), "exit_reason_group"] = "UNKNOWN"
    def aggregate(keys: list[str]) -> pd.DataFrame:
        out = x.groupby(keys, dropna=False, as_index=False).agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            mean_net_pnl=("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda s: float((s > 0).mean() * 100)),
            loss_rows=("net_pnl", lambda s: int((s < 0).sum())),
            gross_wins=("net_pnl", lambda s: float(s[s > 0].sum())),
            gross_losses=("net_pnl", lambda s: float(s[s < 0].sum())),
        )
        out["profit_factor"] = out.apply(lambda r: r["gross_wins"] / abs(r["gross_losses"]) if r["gross_losses"] < 0 else (None if r["gross_wins"] == 0 else "NO_LOSSES_IN_GROUP"), axis=1)
        return out.sort_values(["net_pnl", "trades"], ascending=[True, False])
    by_symbol = aggregate(["symbol"])
    by_reason = aggregate(["exit_reason_group"])
    summary = {
        "status": "SCORED" if len(x) >= 20 else "SMALL_SAMPLE",
        "executed_trade_rows": int(len(x)),
        "symbols": int(x["symbol"].nunique()),
        "net_pnl_sum": float(x["net_pnl"].sum()),
        "loss_making_symbols": int((by_symbol["net_pnl"] < 0).sum()),
        "worst_symbol": str(by_symbol.iloc[0]["symbol"]) if not by_symbol.empty else None,
        "worst_symbol_net_pnl": float(by_symbol.iloc[0]["net_pnl"]) if not by_symbol.empty else None,
        "exit_reason_field_used": reason_col,
        "diagnostic_only": True,
    }
    return by_symbol, by_reason, summary


def _v2_rejection_diagnostics(v2: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize rejected candidate outcomes and surface high-confidence counterfactuals."""
    required = {"target_date", "symbol", "signal", "reference_price"}
    actual_required = {"date", "symbol", "close"}
    if v2.empty or not required.issubset(v2.columns) or history.empty or not actual_required.issubset(history.columns):
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT_DATA", "rows": 0}
    x = v2.copy()
    x["target_date"] = pd.to_datetime(x["target_date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["reference_price"] = pd.to_numeric(x["reference_price"], errors="coerce")
    x["signal_norm"] = x["signal"].astype(str).str.upper().str.strip()
    reason_col = next((c for c in ("no_trade_reason", "phase4_reject_reason") if c in x), None)
    if reason_col:
        primary = x[reason_col].fillna("").astype(str).str.strip()
    else:
        primary = pd.Series("", index=x.index)
    if "phase4_reject_reason" in x and reason_col != "phase4_reject_reason":
        secondary = x["phase4_reject_reason"].fillna("").astype(str).str.strip()
        primary = primary.where(primary.ne(""), secondary)
    x["rejection_reason"] = primary.where(primary.ne(""), "NOT_RECORDED")
    x.loc[x["signal_norm"].eq("BUY"), "rejection_reason"] = "SELECTED_BUY"
    h = history[["date", "symbol", "close"]].copy()
    h["target_date"] = pd.to_datetime(h.pop("date"), errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    h["actual_close"] = pd.to_numeric(h.pop("close"), errors="coerce")
    h = h.dropna(subset=["target_date", "symbol"]).drop_duplicates(["target_date", "symbol"], keep="last")
    x = x.merge(h, on=["target_date", "symbol"], how="left", validate="many_to_one")
    x["counterfactual_close_return_pct"] = (x["actual_close"] / x["reference_price"] - 1) * 100
    x["positive_counterfactual"] = x["counterfactual_close_return_pct"] > 0
    conf_col = next((c for c in ("confidence_v3", "confidence_score", "confidence_pct") if c in x), None)
    x["confidence_value"] = pd.to_numeric(x[conf_col], errors="coerce") if conf_col else np.nan
    skipped = x.loc[~x["signal_norm"].eq("BUY") & x["counterfactual_close_return_pct"].notna()].copy()
    if not skipped.empty and skipped["confidence_value"].notna().any():
        ranks = skipped["confidence_value"].rank(method="average", pct=True)
        skipped["confidence_bucket"] = pd.cut(ranks, [0, .2, .4, .6, .8, 1.000001], labels=["Q1_LOW", "Q2", "Q3", "Q4", "Q5_HIGH"], include_lowest=True).astype(str)
    else:
        skipped["confidence_bucket"] = "UNKNOWN"
    by_reason = skipped.groupby("rejection_reason", as_index=False, dropna=False).agg(
        rows=("counterfactual_close_return_pct", "size"),
        mean_close_return_pct=("counterfactual_close_return_pct", "mean"),
        median_close_return_pct=("counterfactual_close_return_pct", "median"),
        positive_close_rate_pct=("positive_counterfactual", lambda s: float(s.mean() * 100)),
        mean_confidence=("confidence_value", "mean"),
    ).sort_values(["rows", "mean_close_return_pct"], ascending=[False, False])
    high = skipped.loc[skipped["confidence_bucket"].eq("Q5_HIGH")].copy()
    high = high.sort_values("counterfactual_close_return_pct", ascending=False)
    summary = {
        "status": "SCORED" if len(skipped) >= 20 else ("SMALL_SAMPLE" if len(skipped) else "INSUFFICIENT_DATA"),
        "skipped_rows_with_actuals": int(len(skipped)),
        "rejection_reasons": int(skipped["rejection_reason"].nunique()),
        "confidence_field_used": conf_col,
        "high_confidence_skips": int(len(high)),
        "high_confidence_mean_close_return_pct": float(high["counterfactual_close_return_pct"].mean()) if len(high) else None,
        "high_confidence_positive_rate_pct": float(high["positive_counterfactual"].mean() * 100) if len(high) else None,
        "counterfactuals_are_not_executable_pnl": True,
        "selection_gates_changed": False,
        "production_model_changed": False,
    }
    keep = [c for c in ("prediction_date", "target_date", "symbol", "rank", "signal", "score", "confidence_value", "confidence_bucket", "rejection_reason", "reference_price", "actual_close", "counterfactual_close_return_pct") if c in high]
    return by_reason, high[keep], summary

def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    scores = _read("phase33_chronological_prediction_scores.csv")
    history = _read("ohlcv.csv")
    panel = _feature_panel(history)

    by_symbol, by_segment, root_summary = _diagnose_forecasts(scores)
    by_symbol.to_csv(DATA / "phase98_forecast_root_cause_by_symbol.csv", index=False)
    by_segment.to_csv(DATA / "phase98_forecast_root_cause_by_segment.csv", index=False)
    (DATA / "phase98_forecast_root_cause_summary.json").write_text(json.dumps(root_summary, indent=2, default=str) + "\n", encoding="utf-8")

    feature_report = panel.copy() if not panel.empty else pd.DataFrame()
    if not feature_report.empty:
        feature_report["feature_date_before_target"] = (
            pd.to_datetime(feature_report["date"]) < pd.to_datetime(feature_report["target_date"])
        )
        has_target = feature_report["target_date"].notna()
        valid_target = feature_report.loc[has_target].copy()
        # Keep this committed artifact compact: aggregated per-symbol feature coverage,
        # never the full row-level OHLCV-derived feature panel.
        summary_columns = [c for c in FEATURES + ["target_return"] if c in valid_target]
        if not valid_target.empty and summary_columns:
            compact = valid_target.groupby("symbol", as_index=False).agg(
                feature_rows=("date", "size"),
                first_feature_date=("date", "min"),
                last_feature_date=("date", "max"),
                **{f"{col}_mean": (col, "mean") for col in summary_columns},
                **{f"{col}_missing_pct": (col, lambda s: float(s.isna().mean() * 100)) for col in summary_columns},
            )
            compact.to_csv(DATA / "phase99_feature_summary_by_symbol.csv", index=False)
        leakage_summary = {
            "status": "SCORED", "feature_rows": int(len(feature_report)),
            "features": FEATURES,
            "feature_date_after_target_rows": int((has_target & ~feature_report["feature_date_before_target"].fillna(False)).sum()),
            "target_rows": int(feature_report["target_return"].notna().sum()),
            "point_in_time_rule": "Features use OHLCV available on feature_date; supervised label is the next observed close return.",
            "row_level_feature_export_written": False,
            "compact_symbol_summary_written": bool(not valid_target.empty),
            "model_or_production_changed": False,
        }
    else:
        leakage_summary = {
            "status": "INSUFFICIENT_DATA", "feature_rows": 0, "features": FEATURES,
            "row_level_feature_export_written": False, "model_or_production_changed": False,
        }
    (DATA / "phase99_feature_integrity_summary.json").write_text(json.dumps(leakage_summary, indent=2, default=str) + "\n", encoding="utf-8")

    candidate_rows, candidate_summary = _walk_forward_candidate(scores, panel)
    candidate_rows.to_csv(DATA / "phase100_walk_forward_challenger.csv", index=False)
    (DATA / "phase100_walk_forward_summary.json").write_text(json.dumps(candidate_summary, indent=2, default=str) + "\n", encoding="utf-8")

    v2 = _read("paper_trades_v2.csv")
    v2_outcomes, v2_summary = _selection_outcomes(v2, history)
    v2_outcomes.to_csv(DATA / "phase101_v2_selection_outcomes.csv", index=False)
    v1_trades = _read("paper_trades.csv")
    v1_daily, v1_summary = _paper_performance(v1_trades, "V1 paper_trades.csv")
    v2_daily, v2_perf_summary = _paper_performance(v2, "V2 paper_trades_v2.csv")
    v1_daily.to_csv(DATA / "phase101_v1_paper_performance_daily.csv", index=False)
    v2_daily.to_csv(DATA / "phase101_v2_paper_performance_daily.csv", index=False)

    v1_by_symbol, v1_by_exit_reason, v1_loss_summary = _v1_loss_attribution(v1_trades)
    v1_by_symbol.to_csv(DATA / "phase102_v1_loss_by_symbol.csv", index=False)
    v1_by_exit_reason.to_csv(DATA / "phase102_v1_loss_by_exit_reason.csv", index=False)
    (DATA / "phase102_v1_loss_attribution_summary.json").write_text(json.dumps(v1_loss_summary, indent=2, default=str) + "\\n", encoding="utf-8")

    v2_by_reason, v2_high_confidence, v2_rejection_summary = _v2_rejection_diagnostics(v2, history)
    v2_by_reason.to_csv(DATA / "phase102_v2_rejection_reason_outcomes.csv", index=False)
    v2_high_confidence.to_csv(DATA / "phase102_v2_high_confidence_skips.csv", index=False)
    (DATA / "phase102_v2_rejection_summary.json").write_text(json.dumps(v2_rejection_summary, indent=2, default=str) + "\\n", encoding="utf-8")

    summary = {
        "phase98_forecast_root_cause": root_summary,
        "phase99_feature_integrity": leakage_summary,
        "phase100_walk_forward_challenger": candidate_summary,
        "phase101_v2_selection_outcomes": v2_summary,
        "phase101_v1_paper_performance": v1_summary,
        "phase101_v2_paper_performance": v2_perf_summary,
        "phase102_v1_loss_attribution": v1_loss_summary,
        "phase102_v2_rejection_diagnostics": v2_rejection_summary,
        "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False, "ranking_or_risk_changed": False,
        "thresholds_changed": False, "automatic_promotion": False,
    }
    (DATA / "phase98_101_quality_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
