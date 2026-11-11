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
- Scored rows: 350
- Holdout metrics: {'rows': 140, 'open_mape_pct': 0.9562807306340388, 'high_mape_pct': 1.4936055054214206, 'low_mape_pct': 1.3839433469388955, 'close_mape_pct': 1.7731354236513803, 'close_direction_accuracy_pct': 48.57142857142857, 'status': 'SCORED'}

## Trading evidence
- Promotion gate: NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE
- Matched sessions: 31; V2 executed trades: 0

## Data and news audits
- Data checks: 6
- News checks: 6

Note: saved-prediction replay is not a model-refit walk-forward backtest. These reports do not change strategy decisions.
