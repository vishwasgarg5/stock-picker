"""Structural regression checks for imported BROKER compatibility schemas.

These schemas are reference contracts. They are not yet enforced against
Stock Picker's legacy persisted CSV/JSON artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "broker_compatibility"
SCHEMA_NAMES = (
    "ohlcv.schema.json",
    "ingestion_manifest.schema.json",
    "universe_observation.schema.json",
    "universe_membership.schema.json",
    "corporate_actions.schema.json",
    "market_calendar.schema.json",
    "fundamentals.schema.json",
    "predictions.schema.json",
    "evaluation.schema.json",
    "paper_trades.schema.json",
)


class BrokerCompatibilitySchemaTests(unittest.TestCase):
    def load_schema(self, name: str) -> dict:
        path = SCHEMA_DIR / name
        self.assertTrue(path.is_file(), f"Missing compatibility schema: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("$schema"), "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(payload.get("type"), "object")
        self.assertEqual(payload.get("properties", {}).get("schema_version", {}).get("const"), "1.0.0")
        self.assertTrue(payload.get("required"), f"{name} must declare required fields")
        return payload

    def test_all_imported_schemas_are_valid_json_objects(self) -> None:
        for name in SCHEMA_NAMES:
            with self.subTest(schema=name):
                self.load_schema(name)

    def test_manifest_requires_hashes_and_research_only_mode(self) -> None:
        schema = self.load_schema("ingestion_manifest.schema.json")
        required = set(schema["required"])
        self.assertTrue({"raw_sha256", "normalized_sha256", "raw_file", "normalized_file"} <= required)
        self.assertEqual(schema["properties"]["research_only"].get("const"), True)

    def test_current_universe_observation_is_never_pit_eligible(self) -> None:
        schema = self.load_schema("universe_observation.schema.json")
        self.assertEqual(schema["properties"]["pit_eligible"].get("const"), False)
        self.assertTrue({"observed_at", "pit_reason"} <= set(schema["required"]))

    def test_historical_membership_requires_effective_and_publication_dates(self) -> None:
        schema = self.load_schema("universe_membership.schema.json")
        self.assertTrue({"effective_from", "published_at", "source"} <= set(schema["required"]))

    def test_corporate_actions_require_source_and_availability_timestamp(self) -> None:
        schema = self.load_schema("corporate_actions.schema.json")
        self.assertTrue({"ex_date", "action_type", "source", "available_at"} <= set(schema["required"]))

    def test_paper_trade_schema_is_explicitly_simulation_only(self) -> None:
        schema = self.load_schema("paper_trades.schema.json")
        required = set(schema["required"])
        self.assertTrue({"fees", "slippage", "decision_at", "fill_at"} <= required)
        self.assertEqual(schema["properties"]["simulation_only"].get("const"), True)


if __name__ == "__main__":
    unittest.main()
