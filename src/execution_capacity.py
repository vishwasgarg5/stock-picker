from __future__ import annotations

"""Scenario-only execution capacity and partial-fill accounting."""
import math
import numpy as np
import pandas as pd


def simulate_capacity_fills(orders: pd.DataFrame, *, max_participation_rate: float,
                            slippage_bps_per_side: float=0.0) -> pd.DataFrame:
    """Apply a one-session volume cap; missing liquidity evidence yields no fill.

    Required fields: symbol, side, requested_quantity, reference_price,
    average_daily_volume_shares. Inputs must be trailing/as-of the order date.
    Slippage is explicitly a scenario, not an observed execution estimate.
    """
    required={"symbol","side","requested_quantity","reference_price","average_daily_volume_shares"}
    missing=required.difference(orders.columns)
    if missing: raise ValueError(f"orders missing columns: {sorted(missing)}")
    if not np.isfinite(max_participation_rate) or not 0<max_participation_rate<=1:
        raise ValueError("max_participation_rate must be in (0, 1]")
    if not np.isfinite(slippage_bps_per_side) or slippage_bps_per_side<0:
        raise ValueError("slippage_bps_per_side must be finite and non-negative")
    x=orders.copy()
    for c in ("requested_quantity","reference_price","average_daily_volume_shares"):
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x["side"]=x.side.astype(str).str.upper().str.strip()
    if not x.side.isin(["BUY","SELL"]).all(): raise ValueError("side must be BUY or SELL")
    if (x.requested_quantity.isna() | ~np.isfinite(x.requested_quantity) | (x.requested_quantity<=0)).any():
        raise ValueError("requested_quantity must be finite and positive")
    if (x.reference_price.isna() | ~np.isfinite(x.reference_price) | (x.reference_price<=0)).any():
        raise ValueError("reference_price must be finite and positive")
    valid=x.average_daily_volume_shares.notna() & np.isfinite(x.average_daily_volume_shares) & (x.average_daily_volume_shares>0)
    x["liquidity_status"]=np.where(valid,"AVAILABLE","UNRESOLVED_NO_FILL")
    x["capacity_quantity"]=np.where(valid,np.floor(x.average_daily_volume_shares*max_participation_rate),0).astype("int64")
    x["filled_quantity"]=np.minimum(np.floor(x.requested_quantity),x.capacity_quantity).astype("int64")
    x["unfilled_quantity"]=(np.floor(x.requested_quantity)-x.filled_quantity).clip(lower=0).astype("int64")
    x["fill_fraction"]=x.filled_quantity/np.floor(x.requested_quantity)
    signed=np.where(x.side.eq("BUY"),1.0,-1.0)
    x["scenario_fill_price"]=x.reference_price*(1+signed*slippage_bps_per_side/10000.0)
    x["filled_notional"]=x.filled_quantity*x.scenario_fill_price
    x["scenario_slippage_rupees"]=x.filled_quantity*(x.scenario_fill_price-x.reference_price)*signed
    x["execution_status"]=np.select([~valid,x.unfilled_quantity.eq(0),x.filled_quantity.gt(0)],
                                    ["BLOCKED_MISSING_LIQUIDITY","FULL_SCENARIO_FILL","PARTIAL_SCENARIO_FILL"],
                                    default="NO_FILL")
    x["slippage_basis"]="SCENARIO_NOT_OBSERVED_EXECUTION"
    return x
