from __future__ import annotations
"""Phase 127: volatility-normalized target experiment; research-only."""
import json
import numpy as np
import pandas as pd
from .pipeline import DATA, FEATURE_COLUMNS, HISTORY_FILE, add_targets, features, _fit_ensemble, _ensemble_predict

OUTPUT_JSON=DATA/"phase127_volatility_target_summary.json"
OUTPUT_CSV=DATA/"phase127_volatility_target_by_fold.csv"

def _mape(pred:np.ndarray,actual:np.ndarray)->float:
    valid=np.isfinite(pred)&np.isfinite(actual)&(actual>-1.0)
    if not valid.any(): raise ValueError("Phase 127 has no finite evaluation rows")
    return float(np.mean(np.abs(pred[valid]-actual[valid])/(1.0+actual[valid]))*100)

def run_volatility_target(frame:pd.DataFrame,n_folds:int=4)->tuple[dict,pd.DataFrame]:
    required={"date","target_date","symbol","target_close_return","volatility20",*FEATURE_COLUMNS}
    missing=sorted(required.difference(frame.columns))
    if missing: raise ValueError(f"Phase 127 input missing columns: {missing}")
    if n_folds<2: raise ValueError("Phase 127 requires at least two folds")
    x=frame.copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce").dt.normalize()
    x["target_date"]=pd.to_datetime(x.target_date,errors="coerce").dt.normalize()
    x["symbol"]=x.symbol.astype(str).str.upper().str.strip()
    # Validate keys before filtering invalid targets/volatility so duplicate observations cannot be hidden.
    keyed=x.dropna(subset=["date","symbol"])
    if keyed.duplicated(["date","symbol"]).any(): raise ValueError("Phase 127 duplicate date/symbol keys")
    for c in FEATURE_COLUMNS+["target_close_return","volatility20"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["date","target_date","symbol","target_close_return","volatility20"])
    x=x.loc[(x.target_date>x.date)&(x.target_close_return>-1.0)&(x.volatility20>0)].copy()
    dates=np.array(sorted(x.date.unique()))
    initial=max(2,int(len(dates)*.60))
    if len(dates)-initial<n_folds: raise ValueError("Insufficient dates for Phase 127 folds")
    blocks=[b for b in np.array_split(dates[initial:],n_folds) if len(b)]
    records=[]
    for fold,block in enumerate(blocks,1):
        start=pd.Timestamp(block[0])
        train=x.loc[x.target_date<start].dropna(subset=FEATURE_COLUMNS).copy()
        test=x.loc[x.date.isin(block)].dropna(subset=FEATURE_COLUMNS).copy()
        if len(train)<500 or len(test)<10: raise ValueError(f"Phase 127 insufficient rows in fold {fold}")
        y_train=train.target_close_return.to_numpy(float)
        y_test=test.target_close_return.to_numpy(float)
        train_scale=np.maximum(train.volatility20.to_numpy(float),1e-4)
        test_scale=np.maximum(test.volatility20.to_numpy(float),1e-4)
        # Only known, trailing volatility at each prediction date is used.
        z_train=y_train/train_scale
        bundle=_fit_ensemble(train[FEATURE_COLUMNS],pd.Series(z_train,index=train.index))
        z_pred,_=_ensemble_predict(bundle,test[FEATURE_COLUMNS])
        normalized_pred=z_pred*test_scale
        # Control is the unchanged raw-return incumbent trained on identical rows.
        control=_fit_ensemble(train[FEATURE_COLUMNS],train.target_close_return)
        control_pred,_=_ensemble_predict(control,test[FEATURE_COLUMNS])
        baseline=np.zeros(len(test))
        base_mape=_mape(baseline,y_test)
        for name,pred in [("incumbent_raw_return_mae",control_pred),("volatility_normalized_target_mae",normalized_pred)]:
            records.append({"fold":fold,"variant":name,
                "train_end_target_date":train.target_date.max().date().isoformat(),
                "test_start_date":test.date.min().date().isoformat(),
                "test_end_date":test.date.max().date().isoformat(),
                "train_rows":len(train),"test_rows":len(test),
                "baseline_mape_pct":base_mape,"model_mape_pct":_mape(pred,y_test),
                "delta_vs_baseline_pp":_mape(pred,y_test)-base_mape,
                "directional_accuracy_pct":float(np.mean(np.sign(pred)==np.sign(y_test))*100),
                "signed_bias_pp":float(np.mean(pred-y_test)*100),
                "mean_abs_predicted_move_pct":float(np.mean(np.abs(pred))*100),
                "mean_abs_realized_move_pct":float(np.mean(np.abs(y_test))*100),
                "forecast_magnitude_ratio":float(np.mean(np.abs(pred))/max(np.mean(np.abs(y_test)),1e-12)),
                "strict_label_cutoff":bool(train.target_date.max()<start)})
    detail=pd.DataFrame(records)
    summaries=[]
    for name,g in detail.groupby("variant",sort=False):
        summaries.append({"variant":name,"mean_fold_mape_pct":float(g.model_mape_pct.mean()),
            "mean_baseline_mape_pct":float(g.baseline_mape_pct.mean()),
            "mean_delta_vs_baseline_pp":float(g.delta_vs_baseline_pp.mean()),
            "folds_beating_baseline":int((g.delta_vs_baseline_pp<0).sum()),
            "folds_tested":len(g),"mean_directional_accuracy_pct":float(g.directional_accuracy_pct.mean()),
            "mean_signed_bias_pp":float(g.signed_bias_pp.mean()),
            "mean_abs_predicted_move_pct":float(g.mean_abs_predicted_move_pct.mean()),
            "mean_abs_realized_move_pct":float(g.mean_abs_realized_move_pct.mean()),
            "mean_forecast_magnitude_ratio":float(g.forecast_magnitude_ratio.mean())})
    challenger=next(v for v in summaries if v["variant"]=="volatility_normalized_target_mae")
    qualified=challenger["mean_delta_vs_baseline_pp"]<0 and challenger["folds_beating_baseline"]>=max(3,int(np.ceil(n_folds*.75)))
    summary={"phase":127,"status":"VOLATILITY_TARGET_AUDIT_COMPLETE","rows":len(x),
        "symbols":int(x.symbol.nunique()),"feature_dates":int(x.date.nunique()),
        "folds_completed":int(detail.fold.nunique()),"strict_no_lookahead":bool(detail.strict_label_cutoff.all()),
        "method":"Expanding chronological folds; training labels strictly precede test block; predict next-session return divided by same-row trailing volatility20; rescale by test-date trailing volatility known at prediction time; raw-return incumbent control and zero-return baseline; no test-label calibration.",
        "variants":summaries,"performance_gate_met":bool(qualified),
        "decision":"RESEARCH_CANDIDATE_ONLY" if qualified else "NO_VOLATILITY_TARGET_QUALIFIED",
        "production":{"champion":"V1","changed":False,"automatic_promotion":False}}
    return summary,detail

def run()->dict:
    history=pd.read_csv(HISTORY_FILE,parse_dates=["date"])
    labelled=add_targets(features(history))
    labelled["target_date"]=labelled.groupby("symbol",sort=False).date.shift(-1)
    summary,detail=run_volatility_target(labelled,n_folds=4)
    OUTPUT_JSON.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    detail.to_csv(OUTPUT_CSV,index=False)
    print(json.dumps(summary,indent=2))
    return summary
if __name__=="__main__": run()
