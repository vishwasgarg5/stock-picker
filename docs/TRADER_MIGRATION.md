# Trader → Stock Picker migration register

**Baseline:** Stock Picker commit \`df5d3c115af06ec8097a6063972627845134a16b\`  
**Migration policy:** adopt useful methodology, not Trader's entire repository or unverified evidence.  
**Production status:** V1 unchanged; V2 remains shadow-only; no promotion, holdout access or live execution is authorized.

## Adopted in this migration

| Trader capability | Stock Picker implementation | Boundary |
|---|---|---|
| Shuffled-label predictive-edge controls | \`src/predictive_edge.py\` | Refits a cloned estimator on shuffled training labels; test labels remain untouched |
| Logistic comparator | \`src/predictive_edge.py\` | Standardized logistic comparator for classification only |
| Regression comparator | \`src/predictive_edge.py\` | Training-mean baseline for continuous price/return targets |
| Dependence-aware uncertainty | \`src/predictive_edge.py\` | Circular moving-block bootstrap over chronological fold-level paired differences |
| Point-in-time feature provenance | \`src/point_in_time_research.py\` | Earliest availability is max(publication time, first-seen time) |
| Security-master structural audit | \`src/point_in_time_research.py\` | Effective-date/identity schema checks; does not certify historical completeness |
| Execution capacity and participation | \`src/execution_capacity.py\` | Explicit one-session volume cap; no invented liquidity threshold |
| Partial/unfilled quantity accounting | \`src/execution_capacity.py\` | Reports requested, filled and unfilled quantities separately |
| Scenario slippage | \`src/execution_capacity.py\` | Explicit caller-supplied basis points; never presented as observed fills |
| Immutable model freeze | \`src/model_freeze.py\` | Write-once artifact manifest binds model, feature/label specs, config, data snapshot and revision |
| Evidence monitoring | \`src/research_monitoring.py\` | Read-only SHA-256 inventory and fail-closed safety-state report |
| Staged validation contract | \`src/staged_validation.py\` | Explicit prerequisites; broad evaluation and confirmatory holdout remain blocked until all pass |
| Five-session ATR triple barrier | \`src/research_labels.py\` | Previously adopted; separate research label only |

## Existing Stock Picker capabilities retained (not duplicated)

Stock Picker already contains model governance and walk-forward evaluation, a 92-file model manifest, portfolio/risk controls, listing-date refresh, drift/monitoring artifacts, execution-cost provenance, broker-document hash verification, and prospective validation. Relevant existing modules include:

- \`src/model_governance.py\`, \`src/advanced_governance.py\`, \`src/walk_forward.py\`
- \`src/risk_management.py\`, \`src/portfolio_comparison.py\`
- \`src/phase119_prospective_validation.py\`
- \`src/phase131_execution_cost_provenance.py\`, \`src/phase132_broker_cost_evidence.py\`

These were not replaced or rewired. The new research modules are opt-in utilities and are not imported by the production pipeline.

## Deliberately not copied verbatim

1. **Trader's Nifty 50 pilot protocol and its 16 pilot-specific gates:** it is a distinct five-name, Nifty 50 TRI-relative experiment with its own registered target and statistical-power assumptions. Copying those constants into Stock Picker would silently redefine its experiment.
2. **Groww-specific fee configuration and pilot slippage thresholds:** broker/account tariffs and numerical liquidity assumptions need independent confirmation for the actual account, instrument, date and execution route. Existing Stock Picker cost evidence remains authoritative.
3. **Trader's broad-universe expansion as an executable universe:** Stock Picker's listing, membership, corporate-action and security-identity evidence must be independently complete first. A schema audit is not proof of a survivorship-free historical universe.
4. **Trader's full portfolio, data acquisition, broker and paper-trading implementation:** Stock Picker already has separate established systems; replacing them wholesale would create unnecessary risk and duplicate ledgers.
5. **Trader's evidence archives, raw reference data and historical reports:** files are not copied merely to create apparent parity. They require source-level review and provenance mapping.

## Operating rules

- Do not call a scenario fill an actual fill. Do not infer fills, costs, volume, or missing observations.
- Do not use current constituents as a historical point-in-time universe.
- Do not run shuffled-label tests on leaked or non-chronological folds.
- Do not interpret predictive AUC/MAE significance as positive net trading edge.
- Do not change the production target, V1, V2, ranking, sizing, paper-trading ledger, promotion gates, or final holdout.
- SHA-256 proves byte identity, not source authenticity or data correctness.
- The new modules have not been executed or tested in this migration, as requested. Test and integration review remain outstanding.

## Retirement condition for Trader

Do not delete Trader until a separate parity review confirms that every capability worth retaining is either implemented and independently verified in Stock Picker or explicitly classified as inapplicable, and no required data, documentation, configuration, workflow, or source dependency remains exclusively in Trader.
