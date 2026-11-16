from pathlib import Path

import pandas as pd

from src.weekend_data_quality_audit import _audit_csv, _audit_json


def test_audit_csv_reports_missing_values_and_duplicates(tmp_path, monkeypatch):
    path = tmp_path / "predictions.csv"
    pd.DataFrame([
        {"symbol": "ABC", "score": 1.0},
        {"symbol": "ABC", "score": None},
    ]).to_csv(path, index=False)

    monkeypatch.setattr("src.weekend_data_quality_audit.ROOT", tmp_path)
    result = _audit_csv(path)

    assert result["status"] == "ISSUES_FOUND"
    assert result["missing_cells"] == 1
    assert result["missing_by_column"] == {"score": 1}
    assert result["duplicate_rows"] == 0
    assert result["scope"] == "MODEL"


def test_audit_json_counts_nulls_recursively(tmp_path, monkeypatch):
    import json
    from src.weekend_data_quality_audit import _audit_json

    path = tmp_path / "model_validation_summary.json"
    path.write_text(json.dumps({"metrics": {"accuracy": 0.8, "loss": None}, "history": [1, None]}))
    monkeypatch.setattr("src.weekend_data_quality_audit.ROOT", tmp_path)

    result = _audit_json(path)

    assert result["status"] == "ISSUES_FOUND"
    assert result["missing_values"] == 2
    assert result["scope"] == "MODEL"
