from __future__ import annotations

"""Dependence-aware research statistics and outcome-purged chronological folds."""
from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np
import pandas as pd


def _moving_block_means(values, *, block_sessions, simulations, rng):
    x=np.asarray(values,dtype=float)
    if block_sessions<1 or block_sessions>len(x): raise ValueError("invalid block_sessions")
    blocks=math.ceil(len(x)/block_sessions)
    starts=rng.integers(0,len(x)-block_sessions+1,size=(simulations,blocks))
    offsets=np.arange(block_sessions)
    sampled=np.empty((simulations,blocks*block_sessions))
    for j in range(blocks):
        sampled[:,j*block_sessions:(j+1)*block_sessions]=x[starts[:,j,None]+offsets]
    return sampled[:,:len(x)].mean(axis=1)


def estimate_precision_lift_mde(daily_counts: pd.DataFrame, *, alpha: float,
                                target_power: float, seed: int,
                                simulations: int=20000, block_sessions: int=20,
                                mde_ceiling: float=0.06) -> dict:
    """Estimate MDE from daily selected precision minus eligible-universe base rate.

    Counts and design parameters are explicit. This is a feasibility estimate,
    not observed predictive evidence. It does not assume Trader's fixed five-pick
    pilot or its minimum-universe size.
    """
    required={"date","selected_positive_count","selected_count","eligible_positive_count","eligible_count"}
    missing=required.difference(daily_counts.columns)
    if missing: raise ValueError(f"power input missing columns: {sorted(missing)}")
    if not (0<alpha<0.5 and 0.5<target_power<1): raise ValueError("invalid alpha or target_power")
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0: raise ValueError("seed must be non-negative integer")
    if simulations<100 or block_sessions<1 or not 0<mde_ceiling<=1: raise ValueError("invalid simulation design")
    x=daily_counts.copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
    if x["date"].isna().any() or x["date"].duplicated().any(): raise ValueError("dates must be valid and unique")
    cols=["selected_positive_count","selected_count","eligible_positive_count","eligible_count"]
    for col in cols:
        v=pd.to_numeric(x[col],errors="coerce")
        if v.isna().any() or not np.isfinite(v).all() or (v%1!=0).any(): raise ValueError(f"{col} must be finite integer counts")
        x[col]=v.astype(int)
    valid=(x.selected_count.gt(0)&x.eligible_count.ge(x.selected_count)&
           x.selected_positive_count.between(0,x.selected_count)&
           x.eligible_positive_count.between(0,x.eligible_count))
    if not valid.all(): raise ValueError("daily counts are inconsistent")
    x=x.sort_values("date",kind="mergesort")
    if len(x)<block_sessions:
        return {"status":"INSUFFICIENT_DATES","qualifying_dates":len(x),"mde":None,
                "decision":"INCONCLUSIVE","promotion_eligible":False}
    lift=(x.selected_positive_count/x.selected_count-
          x.eligible_positive_count/x.eligible_count).to_numpy(dtype=float)
    rng=np.random.default_rng(int(seed))
    null=_moving_block_means(lift-lift.mean(),block_sessions=block_sessions,
                             simulations=simulations,rng=rng)
    critical=float(np.quantile(null,1-alpha))
    grid=np.linspace(0,.5,1001)
    power=np.array([(null+effect>critical).mean() for effect in grid])
    found=np.flatnonzero(power>=target_power)
    mde=float(grid[found[0]]) if len(found) else None
    return {"status":"ESTIMATED_FEASIBILITY_ONLY","qualifying_dates":len(x),
            "mean_observed_precision_lift":float(lift.mean()),"alpha_one_sided":alpha,
            "target_power":target_power,"seed":int(seed),"simulations":simulations,
            "block_sessions":block_sessions,"mde":mde,"mde_ceiling":mde_ceiling,
            "decision":"INCONCLUSIVE" if mde is None or mde>mde_ceiling else "MDE_WITHIN_REGISTERED_CEILING",
            "method":"centered_moving_time_block_bootstrap","promotion_eligible":False}


def newey_west_mean_inference(values, *, max_lags: int) -> dict:
    """HAC/Newey-West inference for a chronological series of daily observations."""
    from scipy.stats import norm
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    n=len(x)
    if n<3 or isinstance(max_lags,bool) or not isinstance(max_lags,int) or not 0<=max_lags<n:
        raise ValueError("need >=3 observations and integer max_lags in [0,n)")
    mean=float(x.mean()); centered=x-mean
    gamma0=float(np.dot(centered,centered)/n)
    long_run=gamma0
    for lag in range(1,max_lags+1):
        gamma=float(np.dot(centered[lag:],centered[:-lag])/n)
        weight=1-lag/(max_lags+1)
        long_run+=2*weight*gamma
    variance=max(long_run,0.0)/n
    se=math.sqrt(variance)
    if se==0:
        z=math.inf if mean>0 else -math.inf if mean<0 else 0.0
        p=0.0 if mean>0 else 1.0 if mean<0 else 1.0
    else:
        z=mean/se
        p=float(norm.sf(z)) # one-sided test of positive mean
    return {"observations":n,"mean":mean,"newey_west_standard_error":se,
            "z_statistic":float(z),"p_value_one_sided_positive":p,
            "max_lags":max_lags,"method":"Newey-West HAC with Bartlett weights",
            "promotion_eligible":False}


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_indices: tuple[int,...]
    test_indices: tuple[int,...]
    test_start: str
    test_end: str
    purge_sessions: int


def build_purged_walk_forward_folds(observations: pd.DataFrame,
                                    development_sessions: Iterable, *,
                                    min_train_sessions: int,
                                    test_window_sessions: int,
                                    purge_sessions: int) -> list[PurgedFold]:
    """Expanding chronological folds purged by actual label/outcome end dates."""
    required={"signal_date","outcome_end_date"}
    missing=required.difference(observations.columns)
    if missing: raise ValueError(f"observations missing columns: {sorted(missing)}")
    sessions=pd.DatetimeIndex(pd.to_datetime(list(development_sessions),errors="coerce")).normalize()
    if sessions.isna().any() or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("development sessions must be valid, unique and chronological")
    if min_train_sessions<1 or test_window_sessions<1 or purge_sessions<1:
        raise ValueError("train, test and purge windows must be positive")
    data=observations.copy().reset_index(drop=True)
    for col in ("signal_date","outcome_end_date"):
        data[col]=pd.to_datetime(data[col],errors="coerce").dt.normalize()
        if data[col].isna().any() or not data[col].isin(sessions).all():
            raise ValueError(f"{col} contains invalid dates outside development calendar")
    lookup={d:i for i,d in enumerate(sessions)}
    signal=data.signal_date.map(lookup).to_numpy(dtype=int)
    outcome=data.outcome_end_date.map(lookup).to_numpy(dtype=int)
    if (outcome<signal).any(): raise ValueError("outcome_end_date precedes signal_date")
    folds=[]; start=min_train_sessions+purge_sessions; number=0
    while start<len(sessions):
        end=min(start+test_window_sessions,len(sessions))
        boundary=start-purge_sessions
        train=np.flatnonzero((signal<start)&(outcome<boundary))
        test=np.flatnonzero((signal>=start)&(signal<end))
        if len(train) and len(test):
            number+=1
            folds.append(PurgedFold(number,tuple(train.tolist()),tuple(test.tolist()),
                                    sessions[start].date().isoformat(),
                                    sessions[end-1].date().isoformat(),purge_sessions))
        start=end
    return folds
