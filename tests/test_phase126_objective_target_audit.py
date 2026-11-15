import numpy as np
import pandas as pd
import pytest
from src.phase126_objective_target_audit import run_objective_audit, _fit_objective
from src.pipeline import FEATURE_COLUMNS

def _frame():
    rng=np.random.default_rng(126)
    dates=pd.bdate_range("2024-01-01",periods=65)
    rows=[]
    for i,date in enumerate(dates[:-1]):
        for j in range(24):
            latent=rng.normal()
            row={"date":date,"target_date":dates[i+1],"symbol":f"S{j:02d}",
                 "target_close_return":0.003*latent+0.02*rng.normal()}
            for col in FEATURE_COLUMNS: row[col]=latent+rng.normal(scale=.8)
            rows.append(row)
    return pd.DataFrame(rows)

def test_objective_audit_has_strict_label_cutoff_and_all_folds():
    summary,detail=run_objective_audit(_frame(),n_folds=4)
    assert summary["folds_completed"]==4
    assert summary["strict_no_lookahead"]
    assert detail["strict_label_cutoff"].all()
    for row in detail.to_dict("records"):
        assert row["train_end_target_date"]<row["test_start_date"]

def test_target_winsorization_uses_training_distribution_only():
    rng=np.random.default_rng(7)
    x=pd.DataFrame(rng.normal(size=(100,3)),columns=["a","b","c"])
    y=pd.Series(rng.normal(size=100))
    bundle=_fit_objective(x,y,"train_only_winsorized_mae")
    clip=bundle["phase126_train_target_clip"]
    assert clip["lower"]==pytest.approx(y.quantile(.005))
    assert clip["upper"]==pytest.approx(y.quantile(.995))

def test_duplicate_date_symbol_keys_fail_closed():
    frame=_frame()
    frame=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="duplicate"):
        run_objective_audit(frame,n_folds=4)

def test_missing_feature_fails_closed():
    with pytest.raises(ValueError,match="missing"):
        run_objective_audit(_frame().drop(columns=[FEATURE_COLUMNS[0]]),n_folds=4)
