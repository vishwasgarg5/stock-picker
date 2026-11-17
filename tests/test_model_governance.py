import json

import numpy as np
import pandas as pd

from src.model_governance import _has_confidence_buckets, _json_default, _target_diagnostics


def test_target_threshold_diagnostics_are_monotonic_in_coverage():
    rows = []
    for i in range(30):
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "symbol": "A",
            "close": 100.0 * (1.01 ** i),
        })
    out = _target_diagnostics(pd.DataFrame(rows))
    assert not out.empty
    assert out["coverage_pct"].is_monotonic_decreasing
    assert 0.15 in out["threshold_pct"].round(2).tolist()


def test_confidence_bucket_evidence_uses_bucket_level_schema():
    summary = pd.DataFrame([{"confidence_promotion_evidence": False}])
    buckets = pd.DataFrame([{"confidence_bucket": "Q1", "rows": 68}])
    assert not _has_confidence_buckets(summary)
    assert _has_confidence_buckets(buckets)


def test_governance_json_serialization_preserves_numpy_boolean_type():
    payload = json.dumps({"passed": np.bool_(True), "count": np.int64(3)}, default=_json_default)
    restored = json.loads(payload)
    assert restored == {"passed": True, "count": 3}
    assert isinstance(restored["passed"], bool)
    assert isinstance(restored["count"], int)
