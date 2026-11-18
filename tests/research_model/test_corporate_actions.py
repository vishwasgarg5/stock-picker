"""Tests for isolated split-adjustment capability."""

import unittest

from research_model.data.corporate_actions import add_split_adjusted_columns


class SplitAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"date": "2024-01-01", "symbol": "AAA", "open": 199, "high": 202, "low": 198, "close": 200, "volume": 1000, "dividends": 0, "stock_splits": 0},
            {"date": "2024-01-02", "symbol": "AAA", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 2000, "dividends": 5, "stock_splits": 2},
            {"date": "2024-01-03", "symbol": "AAA", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 2100, "dividends": 0, "stock_splits": 0},
        ]

    def test_rebases_only_pre_split_rows_and_preserves_raw_input(self):
        original = [dict(row) for row in self.rows]
        result = add_split_adjusted_columns(self.rows)
        self.assertEqual(result[0]["close"], 200)
        self.assertEqual(result[0]["split_adjusted_close"], 100)
        self.assertEqual(result[1]["split_adjusted_close"], 100)
        self.assertEqual(result[0]["split_adjusted_volume"], 2000)
        self.assertEqual(result[1]["split_adjusted_volume"], 2000)
        self.assertEqual(result[1]["dividends"], 5)
        self.assertEqual(self.rows, original)

    def test_multiple_splits_compound_for_earlier_history(self):
        rows = [
            {"date": "2023-01-01", "symbol": "AAA", "open": 400, "high": 400, "low": 400, "close": 400, "volume": 100},
            {"date": "2023-06-01", "symbol": "AAA", "open": 200, "high": 200, "low": 200, "close": 200, "volume": 200, "stock_splits": 2},
            {"date": "2024-01-01", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 400, "stock_splits": 2},
        ]
        result = add_split_adjusted_columns(rows)
        self.assertEqual(result[0]["split_adjusted_close"], 100)
        self.assertEqual(result[0]["split_adjusted_volume"], 400)
        self.assertEqual(result[1]["split_adjusted_close"], 100)
        self.assertEqual(result[1]["split_adjusted_volume"], 400)

    def test_symbols_are_adjusted_independently_and_output_order_is_preserved(self):
        rows = [
            {"date": "2024-01-01", "symbol": "AAA", "open": 200, "high": 200, "low": 200, "close": 200, "volume": 100},
            {"date": "2024-01-01", "symbol": "BBB", "open": 200, "high": 200, "low": 200, "close": 200, "volume": 100},
            {"date": "2024-01-02", "symbol": "AAA", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 200, "stock_splits": 2},
        ]
        result = add_split_adjusted_columns(rows)
        self.assertEqual([r["symbol"] for r in result], ["AAA", "BBB", "AAA"])
        self.assertEqual(result[0]["split_adjusted_close"], 100)
        self.assertEqual(result[1]["split_adjusted_close"], 200)

    def test_invalid_split_ratio_and_duplicate_keys_fail_closed(self):
        bad = [dict(row) for row in self.rows]
        bad[1]["stock_splits"] = -2
        with self.assertRaisesRegex(ValueError, "negative stock_splits"):
            add_split_adjusted_columns(bad)
        duplicate = [dict(self.rows[0]), dict(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "duplicate symbol/date"):
            add_split_adjusted_columns(duplicate)


if __name__ == "__main__":
    unittest.main()
