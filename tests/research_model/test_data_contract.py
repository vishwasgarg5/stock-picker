"""Unit tests for the isolated research input contract."""

import unittest
from datetime import date, datetime

from research_model.data.partition_keys import partition_key, year_quarter
from research_model.data.validate_ohlcv import validate_ohlcv_row


class PartitionKeyTests(unittest.TestCase):
    def test_quarter_boundaries(self):
        cases = {
            date(2020, 1, 1): "year=2020/quarter=Q1",
            date(2020, 3, 31): "year=2020/quarter=Q1",
            date(2020, 4, 1): "year=2020/quarter=Q2",
            date(2020, 6, 30): "year=2020/quarter=Q2",
            date(2020, 7, 1): "year=2020/quarter=Q3",
            date(2020, 10, 1): "year=2020/quarter=Q4",
            date(2020, 12, 31): "year=2020/quarter=Q4",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(partition_key(value), expected)

    def test_datetime_is_supported(self):
        self.assertEqual(
            year_quarter(datetime(2024, 12, 31, 23, 59)),
            (2024, 4),
        )


class OhlcvValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid_row = {
            "date": "2024-01-02",
            "symbol": "RELIANCE",
            "open": 100,
            "high": 110,
            "low": 95,
            "close": 105,
            "volume": 1000,
        }

    def test_valid_row_has_no_errors(self):
        self.assertEqual(validate_ohlcv_row(self.valid_row), [])

    def test_invalid_high_is_reported(self):
        row = {**self.valid_row, "high": 99}
        self.assertIn(
            "high is below another OHLC value",
            validate_ohlcv_row(row),
        )

    def test_invalid_volume_is_reported(self):
        row = {**self.valid_row, "volume": -1}
        self.assertIn("volume is negative", validate_ohlcv_row(row))

    def test_missing_columns_are_reported(self):
        row = {"date": "2024-01-02"}
        self.assertTrue(any("missing required column" in x for x in validate_ohlcv_row(row)))


if __name__ == "__main__":
    unittest.main()
