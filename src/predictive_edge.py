from __future__ import annotations

"""Research-only predictive-edge diagnostics; never a promotion or deployment gate."""
import math
import numpy as np


def circular_block_bootstrap_mean(values, *, block_length=2, repeats=5000, seed=42061):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    if x.size < 2 or isinstance(block_length,bool) or not 1 <= block_length <= x.size:
        raise ValueError("need at least two finite observations and a valid block_length")
    if isinstance(repeats,bool) or not isinstance(repeats,int) or repeats < 100:
        raise ValueError("repeats must be an integer >= 100")
    rng=np.random.default_rng(seed)
    blocks=int(math.ceil(x.size/block_length))
    offsets=np.arange(block_length)
    estimates=np.empty(repeats)
    for i in range(repeats):
        starts=rng.integers(0,x.size,size=blocks)
        sample=np.concatenate([x[(s+offsets)%x.size] for s in starts])[:x.size]
        estimates[i]=sample.mean()
    return {"method":"circular_moving_block_bootstrap","block_length":int(block_length),
            "repeats":int(repeats),"confidence_level":0.95,
            "lower":float(np.quantile(estimates,.025)),
            "upper":float(np.quantile(estimates,.975)),
            "mean":float(x.mean())}


def run_shuffled_label_control(estimator, X_train, y_train, X_test, y_test, *,
                               task, repeats=500, seed=42061):
    """Refit a fresh estimator on shuffled TRAIN labels; test labels stay untouched."""
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    if task not in {"classification","regression"}:
        raise ValueError("task must be classification or regression")
    if isinstance(repeats,bool) or not isinstance(repeats,int) or repeats<100:
        raise ValueError("repeats must be an integer >= 100")
    y=np.asarray(y_train)
    yt=np.asarray(y_test)
    if len(y)!=len(X_train) or len(yt)!=len(X_test) or len(y)==0 or len(yt)==0:
        raise ValueError("features and labels must have matching non-empty rows")
    if task=="classification" and (np.unique(y).size<2 or np.unique(yt).size<2):
        raise ValueError("classification shuffle control requires both classes in train and test")
    rng=np.random.default_rng(seed)
    scores=[]
    for _ in range(repeats):
        shuffled=rng.permutation(y)
        model=clone(estimator)
        model.fit(X_train,shuffled)
        if task=="classification":
            if hasattr(model,"predict_proba"):
                pred=model.predict_proba(X_test)[:,1]
            elif hasattr(model,"decision_function"):
                pred=model.decision_function(X_test)
            else:
                raise TypeError("classifier must expose predict_proba or decision_function")
            score=float(roc_auc_score(yt,pred))
        else:
            score=float(mean_absolute_error(yt,model.predict(X_test)))
        if np.isfinite(score): scores.append(score)
    if len(scores)<repeats:
        raise ValueError("non-finite shuffled-label metric; control is incomplete")
    better=(lambda v: v>=0) if task=="classification" else (lambda v: v<=0)
    return {"task":task,"metric":"roc_auc" if task=="classification" else "mae",
            "repeats":repeats,"seed":seed,"scores":scores,
            "mean":float(np.mean(scores)),
            "q05":float(np.quantile(scores,.05)),
            "q95":float(np.quantile(scores,.95)),
            "status":"COMPLETED",
            "guardrail":"Only training labels are permuted. This is a null diagnostic, not a promotion criterion."}


def logistic_regression_auc(X_train, y_train, X_test, y_test):
    """Fit a standardized logistic comparator on training data only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42061))
    model.fit(X_train,y_train)
    if len(np.unique(y_test))<2: raise ValueError("test fold must contain both classes")
    return {"model":"standardized_logistic_regression","test_auc":float(roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))}


def mean_regression_baseline(y_train, y_test):
    """Return out-of-sample MAE for a training-mean baseline."""
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error
    model=DummyRegressor(strategy="mean").fit(np.zeros((len(y_train),1)),y_train)
    pred=model.predict(np.zeros((len(y_test),1)))
    return {"model":"training_mean_dummy_regressor","test_mae":float(mean_absolute_error(y_test,pred))}


def summarize_classification_edge(fold_metrics, *, repeats=5000, seed=42061):
    """Summarize chronological-fold AUCs and independently computed shuffle controls."""
    rows=[]
    for fold in fold_metrics.get("folds",[]):
        model=fold.get("model_auc"); baseline=fold.get("logistic_auc")
        null=[float(v) for v in fold.get("shuffled_label_auc",[]) if v is not None and np.isfinite(v)]
        p=None if model is None or not null else float((1+sum(v>=model for v in null))/(1+len(null)))
        rows.append({"fold":fold.get("fold"),"model_auc":model,"logistic_auc":baseline,
                     "paired_auc_difference":None if model is None or baseline is None else float(model-baseline),
                     "shuffle_p_value":p,"shuffle_runs":len(null),
                     "shuffle_q95":None if not null else float(np.quantile(null,.95))})
    diffs=[r["paired_auc_difference"] for r in rows if r["paired_auc_difference"] is not None and np.isfinite(r["paired_auc_difference"])]
    ci=circular_block_bootstrap_mean(diffs,block_length=min(2,len(diffs)),repeats=repeats,seed=seed) if len(diffs)>=4 else None
    pvals=[r["shuffle_p_value"] for r in rows if r["shuffle_p_value"] is not None]
    return {"status":"DIAGNOSTIC_ONLY","folds":rows,"fold_count":len(rows),
            "mean_auc_difference":None if not diffs else float(np.mean(diffs)),
            "paired_difference_95pct_ci":ci,
            "folds_model_above_shuffle_q95":sum(r["model_auc"] is not None and r["shuffle_q95"] is not None and r["model_auc"]>r["shuffle_q95"] for r in rows),
            "mean_shuffle_p_value":None if not pvals else float(np.mean(pvals)),
            "guardrail":"AUC evidence is not net-return evidence and cannot authorize promotion."}


def summarize_regression_edge(fold_metrics, *, repeats=5000, seed=42061):
    """Compare chronological-fold MAE with a training-only baseline and shuffled-label null."""
    rows=[]
    for fold in fold_metrics.get("folds",[]):
        model=fold.get("model_mae"); baseline=fold.get("baseline_mae")
        null=[float(v) for v in fold.get("shuffled_label_mae",[]) if v is not None and np.isfinite(v)]
        p=None if model is None or not null else float((1+sum(v<=model for v in null))/(1+len(null)))
        rows.append({"fold":fold.get("fold"),"model_mae":model,"baseline_mae":baseline,
                     "baseline_minus_model_mae":None if model is None or baseline is None else float(baseline-model),
                     "shuffle_lower_tail_p_value":p,"shuffle_runs":len(null),
                     "shuffle_q05":None if not null else float(np.quantile(null,.05))})
    diffs=[r["baseline_minus_model_mae"] for r in rows if r["baseline_minus_model_mae"] is not None and np.isfinite(r["baseline_minus_model_mae"])]
    ci=circular_block_bootstrap_mean(diffs,block_length=min(2,len(diffs)),repeats=repeats,seed=seed) if len(diffs)>=4 else None
    return {"status":"DIAGNOSTIC_ONLY","folds":rows,"fold_count":len(rows),
            "mean_baseline_minus_model_mae":None if not diffs else float(np.mean(diffs)),
            "paired_difference_95pct_ci":ci,
            "guardrail":"Lower MAE is better. Use only chronological out-of-sample folds; no result authorizes promotion."}
