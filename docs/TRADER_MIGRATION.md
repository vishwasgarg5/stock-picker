# Trader → Stock Picker migration register

This register tracks capability transfer, not file-count parity. Trader contains
large evidence archives and source artifacts that must not be copied wholesale.

## Batch 1 — separate five-session research labels

Transferred to \`src/research_labels.py\` with independent tests. The function
uses next-session-open entries, ATR frozen at signal close, explicit gap exits,
ambiguous same-day double-touch handling, and unresolved incomplete horizons.

This is a research-only label generator. It is not imported by the production
pipeline and does not change the existing next-session OHLC target, V1, V2,
ranking, sizing, paper-trading ledger, promotion gates, or final holdout.

No model has been trained and no predictive-performance claim is made. Before
using the labels in an experiment, register its universe, point-in-time data
requirements, label overlap/purge rules, costs, and chronological validation.

## Capability map

| Trader capability | Stock Picker disposition | Status |
|---|---|---|
| Staged model validation and locked holdout | Existing Stock Picker model governance and walk-forward controls; compare protocol details before changing anything | Audit required |
| Predictive-edge AUC/shuffled-label controls | Trader implementation is classification-specific; Stock Picker's current walk-forward target is regression/price forecasting | Adaptation required; do not wire in blindly |
| Triple-barrier labels | Separate research-only module | Batch 1 transferred |
| Liquidity/participation/partial-fill pilot | Existing portfolio, risk, paper-trading, and execution-cost evidence controls | Gap audit required |
| Hash-based monitoring | Existing 92-file SHA-256 model manifest and audit reports | Provenance integration audit required |
| Point-in-time universe | Stock Picker has listing, membership, and historical coverage workflows; Trader's broader universe is itself data-blocked | No universe expansion authorized |

## Migration acceptance

A capability is complete only after source review, tests, integration where
appropriate, and evidence that its use does not weaken Stock Picker safeguards.
Do not delete Trader based on code presence alone. Preserve Trader until all
valuable capabilities have an equivalent, verified implementation or have been
formally classified as inapplicable.
