"""Tests for independent partition archive integrity verification."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_model.data.partition_ohlcv import partition_ohlcv
from research_model.data.verify_partitions import verify_partition_archive


class VerifyPartitionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        self.archive = self.root / "archive"
        rows = [
            {"date": "2024-04-02", "symbol": "BBB", "open": "20", "high": "22", "low": "19", "close": "21", "volume": "200", "dividends": "0", "stock_splits": "0"},
            {"date": "2024-03-28", "symbol": "AAA", "open": "10", "high": "11", "low": "9", "close": "10.5", "volume": "100", "dividends": "0", "stock_splits": "0"},
            {"date": "2024-04-02", "symbol": "AAA", "open": "30", "high": "32", "low": "29", "close": "31", "volume": "300", "dividends": "0", "stock_splits": "0"},
        ]
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        partition_ohlcv(self.source, self.archive, apply=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_archive_passes_and_reports_aggregates(self):
        result = verify_partition_archive(self.archive)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["partition_count"], 2)
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["symbol_count"], 2)

    def test_modified_partition_fails_checksum(self):
        path = self.archive / "partitioned/ohlcv/year=2024/quarter=Q2/part-000.csv"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_partition_archive(self.archive)

    def test_manifest_count_tampering_is_detected(self):
        path = self.archive / "manifests/partition_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["source"]["row_count"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "aggregate row count"):
            verify_partition_archive(self.archive)

    def test_missing_partition_fails(self):
        path = self.archive / "partitioned/ohlcv/year=2024/quarter=Q1/part-000.csv"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_partition_archive(self.archive)


if __name__ == "__main__":
    unittest.main()
