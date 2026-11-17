from __future__ import annotations

"""Read-only research evidence monitor; never changes promotion or trading state."""
import hashlib
from pathlib import Path


def build_research_monitor_report(*, artifact_paths, repository_revision,
                                  promotion_eligible=False, live_execution_enabled=False,
                                  final_holdout_unlocked=False) -> dict:
    items=[]
    for raw in artifact_paths:
        path=Path(raw)
        if not path.is_file():
            items.append({"path":str(path),"status":"MISSING","sha256":None,"bytes":None})
            continue
        h=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        items.append({"path":str(path),"status":"PRESENT","sha256":h.hexdigest(),"bytes":path.stat().st_size})
    blockers=[x["path"] for x in items if x["status"]!="PRESENT"]
    safe=(promotion_eligible is False and live_execution_enabled is False and final_holdout_unlocked is False)
    return {"schema_version":1,"repository_revision":str(repository_revision),
            "artifacts":items,"missing_artifacts":blockers,
            "safety_state":{"promotion_eligible":promotion_eligible,
                            "live_execution_enabled":live_execution_enabled,
                            "final_holdout_unlocked":final_holdout_unlocked},
            "status":"EVIDENCE_INVENTORY_ONLY" if safe and not blockers else "BLOCKED_OR_UNSAFE",
            "report_is_read_only":True,
            "guardrail":"Monitoring does not certify source truth, model quality, readiness, or execution fills; it cannot alter promotion or holdout state."}
