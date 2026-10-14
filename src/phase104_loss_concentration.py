from __future__ import annotations

"""Phase 104: V1 paper-loss concentration and repeatability audit.

Uses already-recorded paper-trade P&L. Descriptive only: no filtering,
position sizing, ranking, risk, or production-model changes.
"""
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def evaluate_loss_concentration(trades: pd.DataFrame, reference_equity: float = 100_000.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "executed_trade_rows": 0,
            "production_model_changed": False, "ranking_or_risk_changed": False,
            "automatic_promotion": False,
        }
    x = trades.copy()
    if "signal" in x:
        x = x[x["signal"].astype(str).str.upper().isin(["BUY", "LONG", "EXECUTED"])].copy()
    pnl_col = next((c for c in ("profit_loss", "net_pnl", "daily_profit_loss") if c in x), None)
    if pnl_col is None or "symbol" not in x:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "executed_trade_rows": int(len(x)),
            "reason": "Missing symbol or recorded P&L column",
            "production_model_changed": False, "ranking_or_risk_changed": False,
            "automatic_promotion": False, "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        }
    x["net_pnl"] = pd.to_numeric(x[pnl_col], errors="coerce")
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x = x.dropna(subset=["net_pnl"])
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "INSUFFICIENT_DATA", "executed_trade_rows": 0,
            "production_model_changed": False, "ranking_or_risk_changed": False,
            "automatic_promotion": False, "production_champion": "V1", "v2_mode": "SHADOW_ONLY",
        }
    date_col = next((c for c in ("target_date", "date", "trade_date", "exit_date") if c in x), None)
    x["session_date"] = pd.to_datetime(x[date_col], errors="coerce").dt.normalize() if date_col else pd.NaT
    reason_col = next((c for c in ("exit_reason", "exit_type", "reason") if c in x), None)
    x["exit_reason_group"] = x[reason_col].fillna("UNKNOWN").astype(str).str.strip() if reason_col else "NOT_RECORDED"
    x.loc[x["exit_reason_group"].eq(""), "exit_reason_group"] = "UNKNOWN"

    def aggregate(key: str) -> pd.DataFrame:
        out = x.groupby(key, dropna=False, as_index=False).agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            mean_net_pnl=("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda s: float((s > 0).mean() * 100)),
            loss_rows=("net_pnl", lambda s: int((s < 0).sum())),
            gross_losses=("net_pnl", lambda s: float(s[s < 0].sum())),
            gross_wins=("net_pnl", lambda s: float(s[s > 0].sum())),
        )
        out["minimum_sample_met_5_trades"] = out["trades"] >= 5
        return out.sort_values(["net_pnl", "trades"], ascending=[True, False]).reset_index(drop=True)

    by_symbol = aggregate("symbol")
    by_reason = aggregate("exit_reason_group")
    daily = x.dropna(subset=["session_date"]).groupby("session_date", as_index=False).agg(
        trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
        losing_trades=("net_pnl", lambda s: int((s < 0).sum())),
        winning_trades=("net_pnl", lambda s: int((s > 0).sum())),
    ).sort_values("session_date").reset_index(drop=True)
    if not daily.empty:
        daily["cumulative_net_pnl"] = daily["net_pnl"].cumsum()
        daily["equity_on_reference"] = float(reference_equity) + daily["cumulative_net_pnl"]
        daily["running_peak_equity"] = daily["equity_on_reference"].cummax().clip(lower=float(reference_equity))
        daily["drawdown_pct_on_reference_equity"] = (
            (daily["equity_on_reference"] / daily["running_peak_equity"]) - 1
        ) * 100
    else:
        daily = pd.DataFrame(columns=["session_date", "trades", "net_pnl", "losing_trades", "winning_trades",
                                      "cumulative_net_pnl", "equity_on_reference", "running_peak_equity",
                                      "drawdown_pct_on_reference_equity"])

    gross_loss = float(-x.loc[x["net_pnl"] < 0, "net_pnl"].sum())
    top3_losses = float(-by_symbol.sort_values("gross_losses").head(3)["gross_losses"].sum()) if not by_symbol.empty else 0.0
    loss_rows = int((x["net_pnl"] < 0).sum())
    summary = {
        "status": "SCORED" if len(x) >= 20 else "SMALL_SAMPLE",
        "executed_trade_rows": int(len(x)),
        "unique_symbols": int(x["symbol"].nunique()),
        "sessions_with_dates": int(x["session_date"].nunique()),
        "net_pnl_sum": float(x["net_pnl"].sum()),
        "winning_trade_rows": int((x["net_pnl"] > 0).sum()),
        "losing_trade_rows": loss_rows,
        "win_rate_pct": float((x["net_pnl"] > 0).mean() * 100),
        "gross_loss_magnitude": gross_loss,
        "top_3_symbols_gross_loss_share_pct": float(top3_losses / gross_loss * 100) if gross_loss > 0 else None,
        "worst_symbol": str(by_symbol.iloc[0]["symbol"]) if not by_symbol.empty else None,
        "worst_symbol_net_pnl": float(by_symbol.iloc[0]["net_pnl"]) if not by_symbol.empty else None,
        "exit_reason_field_used": reason_col,
        "net_pnl_source_field": pnl_col,
        "costs_subtracted_again": False,
        "minimum_sample_for_symbol_level_action": 5,
        "symbol_rows_below_minimum_sample": int((by_symbol["trades"] < 5).sum()) if not by_symbol.empty else 0,
        "max_drawdown_pct_on_reference_equity": float(daily["drawdown_pct_on_reference_equity"].min()) if not daily.empty else None,
        "diagnostic_only": True,
        "production_champion": "V1",
        "v2_mode": "SHADOW_ONLY",
        "production_model_changed": False,
        "ranking_or_risk_changed": False,
        "thresholds_changed": False,
        "automatic_promotion": False,
        "interpretation": "Loss concentration is descriptive. Do not blacklist symbols or alter risk based on small samples; recorded P&L is used as-is and costs are not subtracted twice.",
    }
    return by_symbol, by_reason, daily, summary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "paper_trades.csv"
    try:
        trades = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        trades = pd.DataFrame()
    by_symbol, by_reason, daily, summary = evaluate_loss_concentration(trades)
    by_symbol.to_csv(DATA / "phase104_v1_loss_concentration_by_symbol.csv", index=False)
    by_reason.to_csv(DATA / "phase104_v1_loss_concentration_by_exit_reason.csv", index=False)
    daily.to_csv(DATA / "phase104_v1_loss_concentration_by_session.csv", index=False)
    (DATA / "phase104_loss_concentration_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
