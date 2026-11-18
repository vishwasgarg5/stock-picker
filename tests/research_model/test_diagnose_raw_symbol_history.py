import csv

from research_model.data.diagnose_raw_symbol_history import diagnose_raw_symbol_history


def write(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_raw_only_symbols_are_profiled_without_being_promoted_or_dropped(tmp_path):
    ohlcv = tmp_path / "ohlcv.csv"
    universe = tmp_path / "universe.csv"
    audit = tmp_path / "coverage.csv"
    write(ohlcv, ["date", "symbol"], [
        {"date": "2020-01-02", "symbol": "OLDCO"},
        {"date": "2021-03-04", "symbol": "OLDCO"},
        {"date": "2022-05-06", "symbol": "CURRENT"},
    ])
    write(universe, ["symbol"], [{"symbol": "CURRENT"}, {"symbol": "DUMMY_PLACEHOLDER"}])
    write(audit, ["symbol"], [{"symbol": "CURRENT"}])

    result = diagnose_raw_symbol_history(ohlcv, universe, audit)
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["raw_row_count"] == 3\n    assert result["current_universe_symbol_count"] == 1
    assert result["raw_only_symbol_count"] == 1
    assert result["raw_only_symbols"] == [{
        "symbol": "OLDCO",
        "row_count": 2,
        "first_date": "2020-01-02",
        "last_date": "2021-03-04",
        "in_current_universe": False,
        "disposition": "REQUIRES_POINT_IN_TIME_HISTORICAL_UNIVERSE_REVIEW",
    }]
    assert result["input_modified"] is False


def test_noncanonical_dates_are_counted_as_diagnostics(tmp_path):
    ohlcv = tmp_path / "ohlcv.csv"
    universe = tmp_path / "universe.csv"
    audit = tmp_path / "coverage.csv"
    write(ohlcv, ["date", "symbol"], [{"date": "2020-01-02T00:00:00", "symbol": "A"}])
    write(universe, ["symbol"], [])
    write(audit, ["symbol"], [])
    result = diagnose_raw_symbol_history(ohlcv, universe, audit)
    assert result["invalid_date_rows"] == 1
    assert result["raw_row_count"] == 0
