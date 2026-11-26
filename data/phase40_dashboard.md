# Stock Picker Phases 32-40 — Diagnostic Dashboard

- Overall status: **REVIEW_REQUIRED**
- Production champion: **V1**
- V2 promotion: **No**
- Production model/ranking/risk changed: **No**

## Selection funnel
- Status: OBSERVED
- Details: {'status': 'OBSERVED', 'stages': 13, 'final_selected_rows': 2}

## Historical prediction replay
- Status: INSUFFICIENT_HOLDOUT
- Scored rows: 120
- Holdout metrics: {'rows': 40, 'open_mape_pct': 0.48835422431031933, 'high_mape_pct': 0.9177378025354677, 'low_mape_pct': 1.0683069910822225, 'close_mape_pct': 1.4782193016260607, 'close_direction_accuracy_pct': 55.00000000000001, 'status': 'SCORED'}

## Trading evidence
- Promotion gate: NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE
- Matched sessions: 37; V2 executed trades: 2

## Data and news audits
- Data checks: 6
- News checks: 6

Note: saved-prediction replay is not a model-refit walk-forward backtest. These reports do not change strategy decisions.
