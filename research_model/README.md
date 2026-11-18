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
