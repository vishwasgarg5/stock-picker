# Stock Picker Phases 32-40 — Diagnostic Dashboard

- Overall status: **REVIEW_REQUIRED**
- Production champion: **V1**
- V2 promotion: **No**
- Production model/ranking/risk changed: **No**

## Selection funnel
- Status: OBSERVED
- Details: {'status': 'OBSERVED', 'stages': 10, 'final_selected_rows': 0}

## Historical prediction replay
- Status: SCORED
- Scored rows: 240
- Holdout metrics: {'rows': 90, 'open_mape_pct': 0.8706060431644448, 'high_mape_pct': 1.6361680911844978, 'low_mape_pct': 1.805162582397586, 'close_mape_pct': 2.1193564362257957, 'close_direction_accuracy_pct': 54.44444444444444, 'status': 'SCORED'}

## Trading evidence
- Promotion gate: NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE
- Matched sessions: 20; V2 executed trades: 0

## Data and news audits
- Data checks: 6
- News checks: 6

Note: saved-prediction replay is not a model-refit walk-forward backtest. These reports do not change strategy decisions.
