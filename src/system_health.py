from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MIN_COVERAGE = 0.95
MAX_PREDICTED_MOVE = 0.40
MIN_PREDICTIONS = 10


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def run_health_check() -> dict:
    universe = _read("universe.csv")
    hist = _read("ohlcv.csv")
    predictions = _read("predictions.csv")
    status = "PASS"
    warnings: list[str] = []
    failures: list[str] = []

    symbols = universe["symbol"].dropna().astype(str).str.upper().str.strip().unique() if "symbol" in universe else []
    latest_date = pd.to_datetime(hist["date"], errors="coerce").max() if "date" in hist else pd.NaT
    latest = hist[pd.to_datetime(hist["date"], errors="coerce").eq(latest_date)] if not hist.empty and pd.notna(latest_date) else pd.DataFrame()
    fresh = latest["symbol"].astype(str).str.upper().str.strip().nunique() if "symbol" in latest else 0
    coverage = fresh / max(len(symbols), 1)

    if coverage < MIN_COVERAGE:
        failures.append(f"OHLCV freshness {coverage:.1%} below {MIN_COVERAGE:.0%}")
    if hist.empty:
        failures.append("OHLCV history is empty")

    pred_ok = True
    if len(predictions) != MIN_PREDICTIONS:
        failures.append(f"prediction count is {len(predictions)}, expected {MIN_PREDICTIONS}")
        pred_ok = False
    if not predictions.empty:
        required = ["symbol", "rank", "base_close", "predicted_open", "predicted_high", "predicted_low", "predicted_close"]
        missing = [c for c in required if c not in predictions.columns]
        if missing:
            failures.append(f"prediction columns missing: {missing}")
            pred_ok = False
        else:
            numeric = predictions[required[1:]].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy()).all() or (numeric <= 0).any().any():
                failures.append("prediction contains non-finite/non-positive values")
                pred_ok = False
            if predictions["symbol"].astype(str).str.upper().duplicated().any():
                failures.append("duplicate prediction symbols")
                pred_ok = False
            if predictions["rank"].nunique() != len(predictions):
                failures.append("prediction ranks are not unique")
                pred_ok = False
            high_bad = predictions["predicted_high"] < predictions[["predicted_open", "predicted_close"]].max(axis=1)
            low_bad = predictions["predicted_low"] > predictions[["predicted_open", "predicted_close"]].min(axis=1)
            if high_bad.any() or low_bad.any():
                failures.append("predicted OHLC consistency failed")
                pred_ok = False
            base = predictions["base_close"].abs().replace(0, np.nan)
            moves = predictions[["predicted_open", "predicted_high", "predicted_low", "predicted_close"]].sub(
                predictions["base_close"], axis=0
            ).abs().div(base, axis=0).max(axis=1)
            if (moves > MAX_PREDICTED_MOVE).any():
                failures.append("predicted move exceeds 40%")
                pred_ok = False

    evaluations = _read("evaluations.csv")
    baseline_note = "no evaluation history yet"
    if not evaluations.empty and "close_mape_pct" in evaluations.columns:
        recent = pd.to_numeric(evaluations["close_mape_pct"], errors="coerce").dropna().tail(5)
        if len(recent) >= 3:
            baseline_note = f"recent close MAPE {recent.mean():.2f}%"
            if "baseline_close_mape_pct" in evaluations.columns:
                base = pd.to_numeric(evaluations["baseline_close_mape_pct"], errors="coerce").dropna().tail(len(recent))
                if len(base) == len(recent) and recent.mean() > base.mean() * 1.10:
                    warnings.append("recent close MAPE is >10% worse than baseline")

    if failures:
        status = "FAILED"
    elif warnings:
        status = "WARNING"

    report = {
        "status": status,
        "universe_symbols": int(len(symbols)),
        "fresh_symbols": int(fresh),
        "coverage_pct": round(coverage * 100, 2),
        "prediction_count": int(len(predictions)),
        "prediction_integrity": "PASS" if pred_ok else "FAIL",
        "latest_data_date": str(latest_date.date()) if pd.notna(latest_date) else "",
        "model_performance": baseline_note,
        "warnings": " | ".join(warnings),
        "failures": " | ".join(failures),
    }
    pd.DataFrame([report]).to_csv(DATA / "system_health.csv", index=False)
    print(
        f"SYSTEM HEALTH: {status} | data={coverage:.1%} | "
        f"predictions={len(predictions)}/{MIN_PREDICTIONS} | {baseline_note}"
    )
    if warnings:
        print("WARNINGS:", *warnings, sep="\n- ")
    if failures:
        print("FAILURES:", *failures, sep="\n- ")

    _send_telegram(report)
    if failures:
        raise RuntimeError("Production health gate failed: " + "; ".join(failures))
    return report


def _send_telegram(report: dict) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    icon = {"PASS": "✅", "WARNING": "⚠️", "FAILED": "❌"}[report["status"]]
    text = (
        f"{icon} STOCK PICKER — SYSTEM HEALTH\n\n"
        f"Status: {report['status']}\n"
        f"Data: {report['fresh_symbols']}/{report['universe_symbols']} ({report['coverage_pct']:.1f}%)\n"
        f"Predictions: {report['prediction_count']}/{MIN_PREDICTIONS}\n"
        f"Integrity: {report['prediction_integrity']}\n"
        f"Performance: {report['model_performance']}\n"
        f"Warnings: {report['warnings'] or 'None'}\n"
        f"Failures: {report['failures'] or 'None'}"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        ).raise_for_status()
    except Exception as exc:
        # Telegram failure must never hide the actual model/data health result.
        print(f"Telegram health notification failed: {exc}")


if __name__ == "__main__":
    run_health_check()
