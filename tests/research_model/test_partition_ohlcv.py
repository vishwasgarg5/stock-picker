"""Tests for deterministic, non-destructive OHLCV partitioning."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research_model.data.partition_ohlcv import partition_ohlcv


class PartitionOhlcvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "ohlcv.csv"
        self.output = self.root / "research_data"
        self.rows = [
            {"date": "2024-04-02", "symbol": "BBB", "open": "20", "high": "22", "low": "19", "close": "21", "volume": "200", "dividends": "0", "stock_splits": "0"},
            {"date": "2024-03-28", "symbol": "AAA", "open": "10", "high": "11", "low": "9", "close": "10.5", "volume": "100", "dividends": "0", "stock_splits": "0"},
            {"date": "2024-04-02", "symbol": "AAA", "open": "30", "high": "32", "low": "29", "close": "31", "volume": "300", "dividends": "0", "stock_splits": "0"},
        ]
        self.columns = list(self.rows[0])
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        self.source_bytes = self.source.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def test_dry_run_validates_and_does_not_write(self):
        result = partition_ohlcv(self.source, self.output)
        self.assertEqual(result["source"]["row_count"], 3)
        self.assertEqual(result["partition_count"], 2)
        self.assertFalse(result["write_applied"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_apply_writes_sorted_partitions_and_manifest(self):
        result = partition_ohlcv(self.source, self.output, apply=True)
        q1 = self.output / "partitioned/ohlcv/year=2024/quarter=Q1/part-000.csv"
        q2 = self.output / "partitioned/ohlcv/year=2024/quarter=Q2/part-000.csv"
        self.assertTrue(q1.is_file())
        self.assertTrue(q2.is_file())
        with q2.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["symbol"] for row in rows], ["AAA", "BBB"])
        manifest_path = self.output / "manifests/partition_manifest.json"
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["source"]["sha256"], hashlib.sha256(self.source_bytes).hexdigest())
        self.assertEqual(stored["partitions"], result["partitions"])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_refuses_existing_output(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            partition_ohlcv(self.source, self.output, apply=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_duplicate_keys_fail_closed(self):
        with self.source.open("a", encoding="utf-8") as handle:
            handle.write(",".join(self.rows[0][column] for column in self.columns) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate symbol/date"):
            partition_ohlcv(self.source, self.output)

    def test_invalid_corporate_action_fails_closed(self):
        self.rows[0]["stock_splits"] = "-2"
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        with self.assertRaisesRegex(ValueError, "stock_splits must be finite and nonnegative"):
            partition_ohlcv(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
