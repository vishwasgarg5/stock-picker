# Isolated Research Model

This package is a clean, capability-oriented research implementation. It is separate from `src/`, does not import the V1 production pipeline, and must not write to existing production data or model paths.

## Build sequence

1. **Input contract** — establish source provenance, OHLCV schema, date coverage, and year/quarter partition keys.
2. **Data quality** — validate dates, symbols, OHLC relationships, duplicates, missing sessions, and corporate actions.
3. **Point-in-time universe** — reconstruct which securities were eligible on each historical date; never apply today's universe to the past.
4. **Feature capabilities** — add one independently tested feature capability per file, only after input integrity is proven.
5. **Labels and splits** — define targets and chronological train/validation/test windows with leakage checks.
6. **Training** — train only on the training window and record data/model manifests.
7. **Prediction and ranking** — generate out-of-sample predictions and deterministic rankings.
8. **Simulation** — paper-trading simulation with costs, liquidity assumptions, and no live orders.
9. **Evaluation** — report year-wise, quarter-wise, regime-wise, and symbol-wise results.
10. **Prospective shadow run** — compare fresh daily evidence without changing V1.

## Historical input layout

The canonical source remains read-only. The research input store will use:

```text
research_data/
  raw/
    ohlcv.csv
  partitioned/
    ohlcv/
      year=YYYY/
        quarter=Q1/
          part-000.csv
        quarter=Q2/
        quarter=Q3/
        quarter=Q4/
  manifests/
    source_manifest.json
    partition_manifest.json
```

Each partition preserves the original `date` and `symbol` columns. Partitioning is a storage convention, not a license to shuffle observations. Never split randomly across time.

## Safety boundaries

- V1 source, V1 outputs, V2 shadow code, and existing historical evidence remain untouched.
- No network access is needed by the research data-validation layer.
- No data partition is generated until source coverage, schema, corporate-action treatment, and provenance have been reviewed.
- No model is considered validated from in-sample metrics.
- Every reported metric must identify its exact date window and input manifest.


## Step 2 — inspect the frozen historical OHLCV source

`research_model.data.inspect_ohlcv` performs a streaming, read-only scan of the committed `data/ohlcv.csv` file. It records the source SHA-256, file size, columns, row count, symbol/date coverage, year-quarter row counts, duplicate symbol/date keys, numeric/price consistency errors, and presence of corporate-action columns. It does not partition or rewrite the source.

Run it from the repository root:

```bash
python -m research_model.data.inspect_ohlcv data/ohlcv.csv --output ohlcv-source-manifest.json
```

The pull-request CI runs the same scan and uploads the full per-symbol manifest as a workflow artifact. The JSON summary is printed in the Actions log.

### Existing audit evidence to reconcile

The repository's existing `data/historical_coverage_audit.json` reports a prior snapshot of 915,478 valid OHLCV rows across 500 symbols, covering 2018-01-01 through 2026-11-17. It marks the audit `REVIEW_REQUIRED`: 497 symbols have verified listing dates, 3 do not, and 6 have data preceding their listing dates. This is inherited audit evidence, not a fresh validation by the new research package.

The new source scan must be reconciled with that evidence before historical partitions are generated. The listing/provenance exceptions and corporate-action treatment remain open data-governance gates. Do not train on these inputs until they are resolved or explicitly excluded under a reviewed policy.


The first full scan on the PR branch found:
- 958,475 rows and 528 symbols;
- date range 2018-01-01 through 2026-11-17;
- zero invalid dates, invalid numeric rows, inconsistent OHLC rows, or duplicate symbol/date keys;
- `dividends` and `stock_splits` are present.

This differs from the prior coverage audit (915,478 rows and a 500-symbol universe). The reconciliation capability reports both values and marks the result `REVIEW_REQUIRED`; it does not rewrite the old audit or assume that the additional rows/symbols are valid for any particular model universe.

The source scan is a structural/data-integrity check only. It does not prove adjusted-price correctness, point-in-time membership, survivorship-bias control, or listing provenance.


## Step 3 — corporate-action treatment

The existing V1 pipeline has a split-only adjustment routine: for each symbol it rebases prices before later stock splits and scales historical volume, while retaining raw execution prices. Its tests cover split-adjusted returns. Dividend cash values are preserved as metadata and are not included in that split adjustment.

The isolated research package now has its own standard-library implementation at `research_model/data/corporate_actions.py`. It:
- retains every raw OHLCV and dividend field;
- adds separate split-adjusted OHLCV fields;
- compounds multiple future split ratios per symbol;
- rejects duplicate symbol/session keys, non-finite or non-positive prices, negative volume, and invalid split ratios;
- does not treat dividends as splits or silently alter prices for dividends.

This is deliberately a split-adjusted price series, **not** a dividend-reinvested total-return series. Dividend-aware labels and returns need a separately specified cash-flow convention and verified action dates/amounts before implementation. The new module is isolated and does not import or modify V1.


## Step 4 — point-in-time membership gate

The committed `data/index_constituents_history.csv` has 994 records across 11 indices and 500 unique symbols, but only one `as_of_date`: 2026-11-17. Its weights are marked `ESTIMATED_FFMC_PROXY`. It is therefore a current snapshot, not a historical membership timeline and not a source of exact historical index weights.

The current `data/universe.csv` has 501 rows including the synthetic `DUMMYHEG` entry. The existing V1 `load_universe` explicitly excludes `DUMMY*` entries, leaving 500 core symbols. The isolated universe reconciliation records the file-level discrepancy; it does not include the dummy symbol as a tradable stock.

The new `research_model.data.inspect_membership` capability labels a single-date file `SNAPSHOT_ONLY_NOT_HISTORICAL_MEMBERSHIP`. Even multiple snapshots are not accepted as continuous membership without explicit effective-from/effective-to intervals. The current evidence cannot support point-in-time index-universe reconstruction or a survivorship-bias-controlled historical backtest.

A reviewed, dated constituent-change source with effective dates is still required. Until then, model training and historical performance claims remain blocked.


## Step 5 — deterministic quarter partition capability

`research_model.data.partition_ohlcv` now provides a fail-closed partition writer. It validates every row before writing, rejects duplicate symbol/session keys and invalid corporate-action values, sorts each partition by date then symbol, and retains every source column and row. It emits a JSON manifest containing the source SHA-256 and size, source row/symbol/date coverage, schema, and per-partition row counts, symbol counts, date ranges, and SHA-256 checksums.

The command is dry-run by default. Writing requires an explicit `--apply` flag and a new output directory whose parent already exists; an existing output directory is never overwritten. Writes are staged in a temporary sibling directory and published by atomic rename. The source file is opened read-only and is never copied over or modified by the command.

Example (plan only):

```bash
python -m research_model.data.partition_ohlcv data/ohlcv.csv --output-root /path/to/new/research_data
```

Example (write only after the data-governance gates are reviewed):

```bash
python -m research_model.data.partition_ohlcv data/ohlcv.csv --output-root /path/to/new/research_data --apply
```

CI tests use small temporary fixtures. CI does not materialize partitions from the repository's historical OHLCV source. The real historical dataset remains unpartitioned until provenance, listing-date exceptions, corporate-action treatment, and point-in-time membership policy have been reviewed.


## Step 6 — independent partition archive verification

`research_model.data.verify_partitions` independently reads the materialized archive and its manifest. It checks the applied-write marker and manifest version, safe expected paths, every partition SHA-256, exact column schema, OHLCV row validity, corporate-action numeric validity, canonical dates, correct year/quarter placement, deterministic date/symbol ordering, duplicate symbol/session keys, per-partition row counts/symbol counts/date ranges, and aggregate source row/symbol/date coverage.

Run it against an archive root after partition creation:

```bash
python -m research_model.data.verify_partitions /path/to/new/research_data
```

The verifier does not modify the archive or source. Its tests deliberately alter a partition, alter manifest counts, and remove a partition to confirm that integrity failures are detected. This verifies internal archive consistency; it does not establish that source prices, listing dates, corporate actions, or historical index membership are economically correct.


## Step 7 — exact source-to-partition reconciliation

`research_model.data.reconcile_partitions` compares the original source CSV with every materialized partition row. It first runs the independent archive verifier, then checks that the source file SHA-256 and schema match the archive manifest. A temporary disk-backed SQLite index matches rows by date/symbol and compares every original CSV field exactly; matched rows are removed from the index, so missing, extra, duplicate, or changed rows fail closed without loading the full source dataset into memory.

Run it after partition creation:

```bash
python -m research_model.data.reconcile_partitions data/ohlcv.csv --archive-root /path/to/new/research_data
```

A `PASS` means exact CSV field-value preservation against the specific source file hash recorded in the manifest. It does not prove that the source itself is economically correct, that corporate actions are complete, or that historical membership is unbiased. Tests use small temporary fixtures only; CI does not create partitions from the committed historical source.


## Step 8 — fail-closed listing-date eligibility

`research_model.data.listing_eligibility` turns the reviewed coverage-audit listing statuses into explicit row-level decisions. Missing listing evidence and unknown listing dates are ineligible. Rows before a verified listing date are held out. Rows on or after a verified listing date may be eligible even when the first observed bar starts later; that gap remains a coverage-review warning, not proof of missing trades. Symbols with a pre-listing anomaly retain a provenance warning even after pre-listing rows are held out.

Run the audit summary:

```bash
python -m research_model.data.listing_eligibility data/historical_coverage_audit.csv
```

This policy does not establish listing dates independently; it consumes the existing reviewed audit. Unknown listing dates remain blocked until authoritative evidence is supplied. It also does not reconstruct historical index membership. A passing listing-date decision is necessary but not sufficient for historical model eligibility.


## Step 9 — embedded corporate-action audit (provenance gate)

`research_model.data.audit_corporate_actions` performs a read-only inventory of the `dividends` and `stock_splits` fields in the raw OHLCV CSV. It reports non-zero event-row counts, unique symbols, date ranges, common values, malformed/negative values, and recognizable provenance columns and their observed values.

The audit is intentionally fail-closed: its status remains `REVIEW_REQUIRED`, and it never describes embedded values as exchange-verified. A populated source/vendor column is metadata, not proof of authenticity. A source file with no event-level provenance fields is explicitly labeled `NO_EVENT_LEVEL_PROVENANCE_FIELDS`. No separate corporate-action event file was found in the repository's tracked `data/` directory during this review.

Run the audit:

```bash
python -m research_model.data.audit_corporate_actions data/ohlcv.csv --output corporate-action-audit.json
```

CI uploads the report with the other historical-data audit artifacts. Until event dates, ratios/amounts, source identity, and adjustment conventions are independently verified, do not use these values to certify split-adjusted prices or dividend-inclusive returns. Negative values are held for review. Dividends remain cash-event metadata; no reinvestment or total-return convention is assumed. The audit does not alter source data or generate adjusted datasets.


The CI audit of the current committed source (958,475 rows, 2018-01-01 through 2026-11-17) observed:
- 4,636 non-zero dividend rows across 432 symbols, from 2018-01-18 through 2026-11-17;
- 179 non-zero split rows across 136 symbols, from 2018-03-15 through 2026-11-04;
- zero malformed or negative values in either embedded action field;
- no event-level provenance columns in the OHLCV source.

These are counts of embedded non-zero values, not counts of verified exchange events. The absence of provenance is the decisive blocker: the current file cannot establish the original source, effective-date semantics, completeness, or whether the values are already adjusted. The CI run therefore succeeds as an audit execution while the report correctly remains `REVIEW_REQUIRED`.

## Step 10 — evidence-linked corporate-action event ledger

The empty template at `research_model/schemas/corporate_action_events.csv` defines a separate event ledger. It intentionally contains a header only; no historical events have been fabricated or inferred from the embedded OHLCV columns.

The validator `research_model.data.validate_action_ledger` checks:
- event identity: symbol, action type, and ex-date, with duplicate keys rejected;
- action-specific fields: positive cash amount plus currency for dividends, or positive integer numerator/denominator for splits;
- separate announcement, ex-, record-, and payment-date fields (the ex-date is the event key; other dates are optional when not evidenced);
- source name, source URL, SHA-256 of the source document, UTC retrieval timestamp, and a stable evidence reference;
- reviewer identity and UTC review timestamp when a record claims `INDEPENDENTLY_VERIFIED`.

For splits, the ratio convention is `new shares / old shares` (numerator / denominator). Dividend amounts must be accompanied by currency. The ledger does not infer missing dates, split ratios, currencies, or source references from OHLCV values.

Run the structural check:

```bash
python -m research_model.data.validate_action_ledger research_model/schemas/corporate_action_events.csv
```

The validator always reports `REVIEW_REQUIRED`, even when every row is structurally valid. A URL, document hash, and self-asserted review status are not proof that the source is authoritative or that an event is correct. Independent human review of the cited source document remains mandatory. This contract does not yet reconcile ledger events against embedded OHLCV values: the source's embedded date and value semantics have not been established, so a comparison would risk a false match or false mismatch. CI validates the empty schema and tests synthetic fixtures only; the actual event ledger remains empty pending authoritative evidence.
