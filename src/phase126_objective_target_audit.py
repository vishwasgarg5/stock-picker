from __future__ import annotations
"""Phase 126: training-objective and next-session target audit; research-only."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from .pipeline import DATA, FEATURE_COLUMNS, HISTORY_FILE, add_targets, features, _fit_ensemble, _ensemble_predict

OUTPUT_JSON = DATA / "phase126_objective_target_summary.json"
OUTPUT_CSV = DATA / "phase126_objective_target_by_fold.csv"

def _metrics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    valid = np.isfinite(predicted) & np.isfinite(actual) & (actual > -1.0)
    p, y = predicted[valid], actual[valid]
    if not len(y):
        raise ValueError("Phase 126 has no finite evaluation rows")
    return {
        "mape_pct": float(np.mean(np.abs(p-y)/(1.0+y))*100.0),
        "directional_accuracy_pct": float(np.mean(np.sign(p)==np.sign(y))*100.0),
        "signed_bias_pp": float(np.mean(p-y)*100.0),
        "mean_abs_predicted_move_pct": float(np.mean(np.abs(p))*100.0),
        "mean_abs_realized_move_pct": float(np.mean(np.abs(y))*100.0),
        "forecast_magnitude_ratio": float(np.mean(np.abs(p))/max(np.mean(np.abs(y)),1e-12)),
    }

def _fit_objective(x: pd.DataFrame, y: pd.Series, variant: str) -> dict:
    if variant == "incumbent_mae":
        return _fit_ensemble(x,y)
    if variant == "squared_error":
        hist = HistGradientBoostingRegressor(loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        extra = ExtraTreesRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=42, criterion="squared_error")
        hist.fit(x,y); extra.fit(x,y)
        return {"models":[hist,extra],"weights":[0.70,0.30],"version":"phase126_squared_error"}
    if variant == "train_only_winsorized_mae":
        lower, upper = y.quantile([0.005,0.995]).to_numpy(dtype=float)
        clipped = y.clip(lower=lower,upper=upper)
        bundle = _fit_ensemble(x,clipped)
        bundle["phase126_train_target_clip"] = {"lower":float(lower),"upper":float(upper),"quantiles":[0.005,0.995]}
        return bundle
    raise ValueError(f"Unknown Phase 126 objective: {variant}")

def run_objective_audit(frame: pd.DataFrame, n_folds: int=4) -> tuple[dict,pd.DataFrame]:
    required={"date","target_date","symbol","target_close_return",*FEATURE_COLUMNS}
    missing=sorted(required.difference(frame.columns))
    if missing: raise ValueError(f"Phase 126 input missing columns: {missing}")
    if n_folds<2: raise ValueError("Phase 126 requires at least two chronological folds")
    x=frame.copy()
    x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
    x["target_date"]=pd.to_datetime(x["target_date"],errors="coerce").dt.normalize()
    x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
    for c in FEATURE_COLUMNS+["target_close_return"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["date","target_date","symbol","target_close_return"])
    x=x.loc[(x["target_date"]>x["date"])&(x["target_close_return"]>-1.0)].copy()
    if x.duplicated(["date","symbol"]).any(): raise ValueError("Phase 126 duplicate date/symbol keys")
    dates=np.array(sorted(x["date"].unique()))
    initial=max(2,int(len(dates)*0.60))
    if len(dates)-initial<n_folds: raise ValueError("Insufficient dates for Phase 126 folds")
    blocks=[b for b in np.array_split(dates[initial:],n_folds) if len(b)]
    rows=[]
    variants=["incumbent_mae","squared_error","train_only_winsorized_mae"]
    for fold,block in enumerate(blocks,1):
        start=pd.Timestamp(block[0])
        train=x.loc[x["target_date"]<start].dropna(subset=FEATURE_COLUMNS).copy()
        test=x.loc[x["date"].isin(block)].dropna(subset=FEATURE_COLUMNS).copy()
        if len(train)<500 or len(test)<10: raise ValueError(f"Phase 126 insufficient rows in fold {fold}")
        y=test["target_close_return"].to_numpy(dtype=float)
        base=_metrics(np.zeros(len(y)),y)
        for variant in variants:
            bundle=_fit_objective(train[FEATURE_COLUMNS],train["target_close_return"],variant)
            pred,_=_ensemble_predict(bundle,test[FEATURE_COLUMNS])
            m=_metrics(pred,y)
            rows.append({"fold":fold,"variant":variant,
                "train_end_target_date":train["target_date"].max().date().isoformat(),
                "test_start_date":test["date"].min().date().isoformat(),
                "test_end_date":test["date"].max().date().isoformat(),
                "train_rows":len(train),"test_rows":len(test),
                "baseline_mape_pct":base["mape_pct"],"model_mape_pct":m["mape_pct"],
                "delta_vs_baseline_pp":m["mape_pct"]-base["mape_pct"],
                **{k:v for k,v in m.items() if k!="mape_pct"},
                "strict_label_cutoff":bool(train["target_date"].max()<start)})
    detail=pd.DataFrame(rows)
    summaries=[]
    for name,g in detail.groupby("variant",sort=False):
        summaries.append({"variant":name,"mean_fold_mape_pct":float(g.model_mape_pct.mean()),
            "mean_baseline_mape_pct":float(g.baseline_mape_pct.mean()),
            "mean_delta_vs_baseline_pp":float(g.delta_vs_baseline_pp.mean()),
            "folds_beating_baseline":int((g.delta_vs_baseline_pp<0).sum()),
            "folds_tested":int(len(g)),
            "mean_directional_accuracy_pct":float(g.directional_accuracy_pct.mean()),
            "mean_signed_bias_pp":float(g.signed_bias_pp.mean()),
            "mean_abs_predicted_move_pct":float(g.mean_abs_predicted_move_pct.mean()),
            "mean_abs_realized_move_pct":float(g.mean_abs_realized_move_pct.mean()),
            "mean_forecast_magnitude_ratio":float(g.forecast_magnitude_ratio.mean())})
    target=x["target_close_return"].to_numpy(dtype=float)
    q=np.quantile(target,[0.001,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.999])
    target_diag={"mean_pct":float(np.mean(target)*100),"std_pct":float(np.std(target)*100),
        "mean_abs_pct":float(np.mean(np.abs(target))*100),"skew":float(pd.Series(target).skew()),
        "kurtosis":float(pd.Series(target).kurtosis()),
        "quantiles_pct":{str(k):float(v*100) for k,v in zip(["0.1%","1%","5%","25%","50%","75%","95%","99%","99.9%"],q)}}
    best=min(summaries,key=lambda z:z["mean_delta_vs_baseline_pp"])
    qualified=best["mean_delta_vs_baseline_pp"]<0 and best["folds_beating_baseline"]>=max(3,int(np.ceil(n_folds*.75)))
    summary={"phase":126,"status":"OBJECTIVE_AUDIT_COMPLETE","rows":len(x),"symbols":int(x.symbol.nunique()),
        "feature_dates":int(x.date.nunique()),"folds_completed":int(detail.fold.nunique()),
        "strict_no_lookahead":bool(detail.strict_label_cutoff.all()),
        "method":"Four expanding chronological folds; labels strictly precede test fold; same feature cohort; compare incumbent MAE ensemble, squared-error ensemble, and MAE trained on training-fold-only 0.5/99.5% winsorized labels; evaluate all on untouched raw test labels against zero-return baseline.",
        "target_distribution":target_diag,"variants":summaries,"best_variant":best["variant"],
        "performance_gate_met":bool(qualified),
        "decision":"RESEARCH_CANDIDATE_ONLY" if qualified else "NO_OBJECTIVE_QUALIFIED",
        "production":{"champion":"V1","changed":False,"automatic_promotion":False}}
    return summary,detail

def run()->dict:
    history=pd.read_csv(HISTORY_FILE,parse_dates=["date"])
    labelled=add_targets(features(history))
    labelled["target_date"]=labelled.groupby("symbol",sort=False)["date"].shift(-1)
    summary,detail=run_objective_audit(labelled,n_folds=4)
    OUTPUT_JSON.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    detail.to_csv(OUTPUT_CSV,index=False)
    print(json.dumps({"phase":126,"rows":summary["rows"],"symbols":summary["symbols"],
        "folds_completed":summary["folds_completed"],"strict_no_lookahead":summary["strict_no_lookahead"],
        "target_distribution":summary["target_distribution"],"variants":summary["variants"],
        "best_variant":summary["best_variant"],"performance_gate_met":summary["performance_gate_met"],
        "outputs":[str(OUTPUT_JSON),str(OUTPUT_CSV)]},indent=2))
    return summary

if __name__=="__main__": run()
