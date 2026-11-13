from __future__ import annotations

import unittest

import pandas as pd

from src.date_utils import normalize_date_column, parse_mixed_dates


class MixedDateParsingTests(unittest.TestCase):
    def test_parses_date_only_and_timestamp_values(self) -> None:
        values = pd.Series([
            "2026-09-24 00:00:00",
            "2026-09-24",
            "2026-09-25T15:30:00",
        ])
        parsed = parse_mixed_dates(values)
        self.assertEqual(parsed.notna().sum(), 3)
        self.assertEqual(parsed.dt.strftime("%Y-%m-%d").tolist(), [
            "2026-09-24", "2026-09-24", "2026-09-25"
        ])

    def test_invalid_values_remain_nat(self) -> None:
        parsed = parse_mixed_dates(pd.Series(["2026-09-24", "not-a-date"]))
        self.assertEqual(parsed.notna().sum(), 1)
        self.assertTrue(pd.isna(parsed.iloc[1]))

    def test_normalize_date_column_preserves_other_columns(self) -> None:
        frame = pd.DataFrame({
            "target_date": ["2026-09-24 00:00:00", "2026-09-24"],
            "symbol": ["ABC", "XYZ"],
        })
        normalized = normalize_date_column(frame, "target_date")
        self.assertEqual(normalized["target_date"].nunique(), 1)
        self.assertEqual(normalized["symbol"].tolist(), ["ABC", "XYZ"])
        self.assertEqual(frame["target_date"].iloc[0], "2026-09-24 00:00:00")


if __name__ == "__main__":
    unittest.main()
