# BROKER → Stock Picker adoption matrix

This document tracks reusable capabilities from `vishwasgarg5/Broker` against the main Stock Picker architecture. Stock Picker remains the canonical project. This is an implementation inventory, not a claim that BROKER is production-ready.

## Operating boundary

- Keep Stock Picker's existing production OHLCV, predictions, paper-trade ledgers, and V1 behaviour unchanged.
- Keep V2 shadow-only; no promotion or live execution.
- Treat BROKER's NSE UDiFF feed as independent research evidence until its completeness and reconciliation are demonstrated.
- Never infer missing historical prices, fees, fills, or effective dates.
- Fail closed when provenance, chronology, or source evidence is missing.

## File-by-file capability mapping

| BROKER path | Capability | Stock Picker counterpart | Status | Adoption decision |
|---|---|---|---|---|
| `src/broker/validation.py` | Strict OHLCV validation, duplicate/session chronology, config checks | `src/broker_data_contracts.py`, existing data-quality audits | **Adopted in part** | Keep the additive contracts; compare edge cases against existing validators and avoid duplicate/conflicting policy. |
| `src/broker/nse_ingest.py` | Official NSE UDiFF EQ bhavcopy download, normalization, raw archive and hash manifest | `src/nse_udiff_reconciliation.py` | **Research-only adoption** | The PR #40 adapter downloads and validates the same official feed, but writes only below `audit/broker_udiff/`; do not silently merge it into canonical OHLCV. |
| `src/broker/data_pipeline.py` | Orchestrates UDiFF ingestion and current Nifty 50 constituent observation | Existing Stock Picker market-data workflow; `src/universe.py` | **Partial / not equivalent** | Add a dated observation feed only if it is useful for reconciliation; current constituents must not be represented as historical membership. |
| `src/broker/readiness.py` | Explicit research gates and next-session chronology | Existing model-promotion/readiness audits; `src/broker_data_contracts.py` | **Adopted in part** | Preserve Stock Picker's stricter gates; never replace them with a smaller generic gate list. |
| `schemas/ohlcv.schema.json` | Versioned normalized OHLCV contract | Existing CSV/data-quality checks and `src/broker_data_contracts.py` | **Runtime parity to verify** | Add schema-driven fixtures/validation only where it does not reject valid legacy Stock Picker records. |
| `schemas/ingestion_manifest.schema.json` | Versioned source, retrieval, file and SHA-256 provenance | `src/broker_data_contracts.py`, `src/nse_udiff_reconciliation.py` | **Research verification implemented** | Persisted raw/normalized bytes are re-hashed from disk and normalized row counts are checked against the manifest; tampering, missing files and count mismatches fail closed. |
| `schemas/universe_observation.schema.json` | Snapshot of observed current universe | `src/nse_universe_observation.py` | **Research-only adoption implemented** | Persist official Nifty 50 observations with raw/normalized hashes and row-count verification under `audit/broker_universe/`; `pit_eligible=false` and no historical membership is inferred. |
| `schemas/universe_membership.schema.json` | Effective-dated point-in-time membership | `src/universe_membership_provenance.py` | **Evidence contract implemented; historical data gap remains** | Strictly validates effective dates, timezone-aware publication timestamps, HTTPS source-document URLs, source-document SHA-256 hashes and non-overlapping intervals. Does not generate membership rows or claim dataset completeness. |
| `schemas/corporate_actions.schema.json` | Source-backed corporate-action event contract | `tests/test_corporate_action_adjustments.py` and existing adjustment pipeline | **Partial** | Align event provenance and availability timestamps; do not replace existing split/dividend handling without regression proof. |
| `schemas/market_calendar.schema.json` | Versioned exchange calendar events | `src/market_calendar.py` | **Existing capability; schema parity to verify** | Retain official NSE calendar and special-session confirmation rules. |
| `schemas/fundamentals.schema.json` | Versioned fundamental snapshots | Existing fundamentals ingestion/history | **Existing capability; schema parity to verify** | Add provenance/availability checks only where missing. |
| `schemas/predictions.schema.json`, `schemas/evaluation.schema.json` | Versioned forecast and realized-outcome contracts | Existing prediction/evaluation artifacts and audits | **Existing capability; schema parity to verify** | Preserve Stock Picker's established keys and no-lookahead checks. |
| `schemas/paper_trades.schema.json` | Versioned paper ledger contract | V1/V2 paper-trade accounting and reconciliation audits | **Existing capability; schema parity to verify** | Do not rewrite historical ledger rows or infer fees to satisfy a schema. |
| `config/*.json` | Central asset, source, feature and risk configuration | Existing source constants, workflow config and model/risk controls | **Selective only** | Do not introduce a second source of truth or change V1 model/risk settings merely to mirror BROKER. |
| `.github/workflows/pre_market.yml`, `post_market.yml`, `ci.yml` | Separate ingestion/validation schedules and CI | Existing daily, weekend, audit and regression workflows | **Existing capability; integration review** | Reuse isolated jobs and artifact reporting where useful; avoid duplicate schedules or external-source load. |

## PR #40 work already present

- `src/broker_data_contracts.py`: normalized OHLCV, chronology/duplicate, timezone-aware point-in-time, safe relative paths, SHA-256 manifests and fail-closed readiness helpers.
- `tests/test_broker_data_contracts.py`: focused contract regression coverage.
- `src/nse_udiff_reconciliation.py`: official NSE UDiFF research ingestion with raw/normalized snapshots, a persisted manifest, and post-write hash/record-count verification under an isolated audit root.
- `src/nse_universe_observation.py`: official Nifty 50 current-membership observations with raw/normalized provenance, hash verification, and an explicit non-PIT boundary under `audit/broker_universe/`.
- `src/universe_membership_provenance.py`: strict validator for externally sourced effective-dated membership records; requires publication timestamps, HTTPS source-document URLs, SHA-256 hashes of original documents, and rejects overlapping intervals. Validation does not assert historical coverage or PIT readiness.
- `tests/test_universe_membership_provenance.py`: nine regression tests for required dates, timestamp timezone, invalid intervals, duplicates, overlap, HTTPS source URLs and SHA-256 source-document hashes.
- `tests/test_nse_universe_observation.py`: deterministic tests for exact 50-symbol snapshots, duplicates, missing/tampered files, and the non-PIT boundary.
- `tests/test_nse_udiff_reconciliation.py`: deterministic ingestion tests.
- Model Progress Audit workflow: runs the contracts and isolated UDiFF probe and reports source status without mutating canonical OHLCV or paper ledgers.

## Remaining implementation sequence

1. **Schema parity:** map BROKER JSON schemas to Stock Picker's actual CSV/JSON fields; add compatibility tests before enforcing any schema at runtime.
2. **Manifest verification:** implemented in PR #40; persisted raw/normalized hashes and normalized record counts are checked from disk, with tampering/missing files rejected. CI evidence is still pending.
3. **Universe observations:** implemented in PR #40 as isolated dated snapshots with raw/normalized hash verification; explicitly non-PIT and never backfilled into historical membership. CI evidence is pending.
4. **Membership provenance:** strict validation contract implemented in PR #40. Official historical source records and completeness reconciliation remain outstanding; current snapshots are not promoted to historical membership.
5. **Corporate-action provenance:** verify source, ex-date, action type, ratio/factor and available-at timestamp while preserving existing adjustment behaviour.
6. **Pipeline integration:** publish these checks in the audit inventory and independent test suite; do not make them production dependencies until evidence and compatibility tests pass.

## Explicitly out of scope

BROKER's README identifies BSE coverage, sector/ETF/Bank Nifty data, historical backfill, corporate actions, official special-session reconciliation, effective-dated historical universe membership and sustained scheduled-run verification as incomplete. These must not be described as implemented features or copied as if production-ready. Live broker connectivity and order execution are excluded permanently from this research-only integration.
