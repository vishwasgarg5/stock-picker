"""Tests for exact source-to-partition reconciliation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_model.data.manifest import sha256_file
from research_model.data.partition_ohlcv import partition_ohlcv
from research_model.data.reconcile_partitions import reconcile_source_to_partitions


class ReconcilePartitionsTests(unittest.TestCase):
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

    def _refresh_partition_checksum(self):
        manifest_path = self.archive / "manifests/partition_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["partitions"]:
            entry["sha256"] = sha256_file(self.archive / entry["path"])
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_exact_reconciliation_passes_for_unsorted_source(self):
        result = reconcile_source_to_partitions(self.source, self.archive)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["source_rows"], 3)
        self.assertEqual(result["partition_rows"], 3)
        self.assertTrue(result["exact_row_match"])

    def test_source_hash_mismatch_fails_before_comparison(self):
        with self.source.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "source SHA-256"):
            reconcile_source_to_partitions(self.source, self.archive)

    def test_rehashed_partition_with_changed_value_fails_exact_match(self):
        path = self.archive / "partitioned/ohlcv/year=2024/quarter=Q2/part-000.csv"
        rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        rows[0]["close"] = "31.25"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self._refresh_partition_checksum()
        with self.assertRaisesRegex(ValueError, "row content differs"):
            reconcile_source_to_partitions(self.source, self.archive)


if __name__ == "__main__":
    unittest.main()
