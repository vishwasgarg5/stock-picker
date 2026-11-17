# Stock Picker Phases 32-40 — Diagnostic Dashboard

- Overall status: **REVIEW_REQUIRED**
- Production champion: **V1**
- V2 promotion: **No**
- Production model/ranking/risk changed: **No**

## Selection funnel
- Status: OBSERVED
- Details: {'status': 'OBSERVED', 'stages': 13, 'final_selected_rows': 2}

## Historical prediction replay
- Status: SCORED
- Scored rows: 380
- Holdout metrics: {'rows': 150, 'open_mape_pct': 0.8649130909582956, 'high_mape_pct': 1.381454621926736, 'low_mape_pct': 1.2474617469339346, 'close_mape_pct': 1.6106374334930635, 'close_direction_accuracy_pct': 52.0, 'status': 'SCORED'}

## Trading evidence
- Promotion gate: NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE
- Matched sessions: 34; V2 executed trades: 2

## Data and news audits
- Data checks: 6
- News checks: 6

Note: saved-prediction replay is not a model-refit walk-forward backtest. These reports do not change strategy decisions.
