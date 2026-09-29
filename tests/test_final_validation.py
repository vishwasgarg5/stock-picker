import pandas as pd

from src.final_validation import run_final_validation


def test_final_validation_holds_when_gates_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.final_validation.DATA", tmp_path)
    monkeypatch.setattr("src.final_validation.OUTPUT", tmp_path / "final_model_validation.csv")

    out = run_final_validation()

    assert bool(out["promotion_gate_passed"].iloc[0]) is False
    assert out["decision"].iloc[0] == "HOLD"
