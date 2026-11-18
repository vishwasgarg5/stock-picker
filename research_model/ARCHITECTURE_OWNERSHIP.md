# Package ownership and migration boundary

## `src/` — production owner

`src/` owns the current production stock-picker lifecycle: scheduled data refresh,
universe maintenance, existing forecasts and rankings, risk controls, paper trading,
production reports, monitoring, and V1/V2 governance. Changes here must preserve
the current production contract unless a separately reviewed migration explicitly
changes it. V1 remains the production baseline; V2 remains shadow-only.

## `research_model/` — research owner

`research_model/` owns the next-generation research contracts and experiments:
immutable raw-data inspection, provenance manifests, historical security identity,
corporate-action evidence, point-in-time universe policies, partitioning, labels,
chronological validation, candidate models, and research-only evaluation.

It must not import from `src/`. Shared behaviour should move into a deliberately
small, neutral module only after its contract is identified and tested; do not make
the research package depend on production internals as a shortcut.

## Data and authority boundaries

- Raw source files remain immutable; derived artifacts are stored separately and
  identify their source manifest hash and producing Git SHA.
- Quarantined symbols/rows remain available for audit but are excluded from
  training and evaluation universes until an explicit eligibility policy admits them.
- Tier 1 permits structural exploration only.
- Tier 2 permits training only on an explicitly reduced, point-in-time eligible
  universe with all included rows passing identity/listing checks.
- Tier 3 is required for promotion claims and requires full source, identity,
  corporate-action, universe, and evaluation evidence.
- No tier authorizes production or live trading by itself. Promotion is a separate
  governance decision.

## Migration rules

1. Keep V1 production behaviour stable while research contracts mature.
2. Migrate duplicated functionality by capability, not by copying modules.
3. Rename phase-numbered modules in small batches: move one capability, update
   imports and workflow references, then run the affected tests before the next batch.
4. Preserve historical phase reports under `data/archive/`; do not delete audit
   history during renames.
5. Record data-manifest hashes, Git SHA, dependency lock identity, and random seeds
   in every model/evaluation report.
