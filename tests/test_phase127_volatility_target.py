import numpy as np
import pandas as pd
import pytest
from src.phase127_volatility_target import run_volatility_target
from src.pipeline import FEATURE_COLUMNS

def _frame():
    rng=np.random.default_rng(127)
    dates=pd.bdate_range("2024-01-01",periods=110)
    rows=[]
    for i,date in enumerate(dates[:-1]):
        for j in range(24):
            z=rng.normal()
            vol=.01+(.02*(j%4))
            row={"date":date,"target_date":dates[i+1],"symbol":f"S{j:02d}",
                 "target_close_return":.002*z+vol*rng.normal(),"volatility20":vol}
            for col in FEATURE_COLUMNS: row[col]=z+rng.normal(scale=.7)
            rows.append(row)
    return pd.DataFrame(rows)

def test_volatility_target_uses_strict_chronological_folds():
    summary,detail=run_volatility_target(_frame(),4)
    assert summary["folds_completed"]==4
    assert summary["strict_no_lookahead"]
    assert detail.strict_label_cutoff.all()
    assert all(r["train_end_target_date"]<r["test_start_date"] for r in detail.to_dict("records"))

def test_duplicate_keys_fail_closed():
    x=_frame(); x=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="duplicate"):
        run_volatility_target(x,4)

def test_nonpositive_volatility_rows_are_excluded():
    x=_frame(); x.loc[x.index[0],"volatility20"]=0
    original_summary,_=run_volatility_target(_frame(),4)
    summary,detail=run_volatility_target(x,4)
    # Feature warm-up and fold eligibility can exclude many rows; assert only
    # that the invalid-volatility observation removes exactly one eligible row.
    assert summary["rows"]==original_summary["rows"]-1

def test_missing_feature_fails_closed():
    with pytest.raises(ValueError,match="missing"):
        run_volatility_target(_frame().drop(columns=[FEATURE_COLUMNS[0]]),4)
