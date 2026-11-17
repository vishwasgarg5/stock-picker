from __future__ import annotations

"""Experimental labels for a separate five-session research track.

This module is intentionally not imported by the production pipeline, V1, V2,
ranking, paper-trading, or model-promotion code.
"""
import numpy as np
import pandas as pd


def triple_barrier_labels(
    prices: pd.DataFrame,
    horizon: int = 5,
    up_multiple: float = 2.0,
    down_multiple: float = 1.5,
) -> pd.DataFrame:
    """Label signal-day rows using next-session-open entries and frozen signal ATR.

    The entry session counts as the first horizon session. Later opening gaps
    execute at the open. A same-session double barrier touch is ambiguous and
    left unlabelled. Rows without a complete forward horizon remain unresolved.
    """
    required = {"open", "high", "low", "close", "atr"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if (not np.isfinite(up_multiple) or not np.isfinite(down_multiple)
            or up_multiple <= 0 or down_multiple <= 0):
        raise ValueError("barrier multiples must be finite and positive")

    data = prices.sort_index()
    if data.index.has_duplicates:
        raise ValueError("Price index must contain unique trading dates")
    ohlc = data[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    values = ohlc.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("OHLC prices must be finite and positive")
    invalid_ranges = (
        (ohlc["high"] < ohlc[["open", "close", "low"]].max(axis=1))
        | (ohlc["low"] > ohlc[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ranges.any():
        raise ValueError("OHLC bars contain inconsistent high/low ranges")
    data = data.copy()
    data[["open", "high", "low", "close"]] = ohlc
    rows: list[dict] = []
    n = len(data)
    for i in range(n):
        atr = float(data["atr"].iloc[i]) if pd.notna(data["atr"].iloc[i]) else np.nan
        result = {
            "target": np.nan,
            "entry_date": data.index[i + 1] if i + 1 < n else pd.NaT,
            "entry_price": np.nan,
            "exit_reason": "unresolved",
            "exit_date": pd.NaT,
            "exit_price": np.nan,
            "outcome_return": np.nan,
        }
        if i + 1 >= n or i + horizon >= n or not np.isfinite(atr) or atr <= 0:
            rows.append(result)
            continue

        entry = float(data["open"].iloc[i + 1])
        if not np.isfinite(entry) or entry <= 0:
            result["exit_reason"] = "invalid_entry"
            rows.append(result)
            continue
        result["entry_price"] = entry
        take_profit = entry + up_multiple * atr
        stop_loss = entry - down_multiple * atr
        resolved = False

        for j in range(i + 1, i + horizon + 1):
            bar = data.iloc[j]
            opn, high, low = (float(bar[k]) for k in ("open", "high", "low"))
            if not np.isfinite([opn, high, low]).all() or opn <= 0:
                result["exit_reason"] = "invalid_bar"
                resolved = True
                break

            if j > i + 1 and opn >= take_profit:
                exit_price, reason = opn, "target_gap"
            elif j > i + 1 and opn <= stop_loss:
                exit_price, reason = opn, "stop_gap"
            elif low <= stop_loss and high >= take_profit:
                result.update({
                    "target": np.nan,
                    "exit_reason": "ambiguous",
                    "exit_date": data.index[j],
                })
                resolved = True
                break
            elif high >= take_profit:
                exit_price, reason = take_profit, "target"
            elif low <= stop_loss:
                exit_price, reason = stop_loss, "stop"
            else:
                continue

            outcome_return = exit_price / entry - 1.0
            result.update({
                "target": int(reason.startswith("target")),
                "exit_reason": reason,
                "exit_date": data.index[j],
                "exit_price": exit_price,
                "outcome_return": outcome_return,
            })
            resolved = True
            break

        if not resolved:
            exit_price = float(data["close"].iloc[i + horizon])
            if not np.isfinite(exit_price) or exit_price <= 0:
                result["exit_reason"] = "invalid_timeout_close"
            else:
                outcome_return = exit_price / entry - 1.0
                result.update({
                    "target": int(outcome_return > 0),
                    "exit_reason": "timeout",
                    "exit_date": data.index[i + horizon],
                    "exit_price": exit_price,
                    "outcome_return": outcome_return,
                })
        rows.append(result)

    return pd.DataFrame(rows, index=data.index)
