from __future__ import annotations

"""Write-once reproducibility manifest for research model versions."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def build_model_freeze(*, model_path, feature_spec_path, label_spec_path,
                       config_path, data_snapshot_path, code_revision,
                       model_version, feature_columns, training_start,
                       training_end, random_seed) -> dict:
    """Bind all prediction-affecting artifacts; hashes do not certify data truth."""
    inputs={"model":model_path,"feature_spec":feature_spec_path,"label_spec":label_spec_path,
            "config":config_path,"data_snapshot":data_snapshot_path}
    hashes={}
    for name,path in inputs.items():
        p=Path(path)
        if not p.is_file(): raise FileNotFoundError(f"{name} artifact missing: {p}")
        hashes[name]={"path":str(p),"sha256":sha256_file(p),"bytes":p.stat().st_size}
    if not code_revision or not model_version or not feature_columns:
        raise ValueError("code_revision, model_version and ordered feature_columns are required")
    if len(set(feature_columns))!=len(feature_columns): raise ValueError("feature_columns must be unique and ordered")
    return {"schema_version":1,"status":"FROZEN_RESEARCH_ARTIFACT",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "model_version":str(model_version),"code_revision":str(code_revision),
            "training_window":{"start":str(training_start),"end":str(training_end)},
            "random_seed":int(random_seed),"feature_columns":list(feature_columns),
            "artifacts":hashes,
            "guardrails":{"production_champion_changed":False,"promotion_authorized":False,
                          "final_holdout_unlocked":False,"live_execution_enabled":False},
            "integrity_note":"SHA-256 proves byte identity only; it does not prove point-in-time correctness, source authenticity, or predictive validity."}


def write_model_freeze(manifest: dict, destination: str | Path) -> Path:
    """Create a new manifest only; never overwrite an existing frozen version."""
    path=Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    fd=os.open(path,flags,0o644)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(manifest,handle,indent=2,sort_keys=True)
            handle.write("\\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path
