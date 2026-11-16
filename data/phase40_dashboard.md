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
- Scored rows: 370
- Holdout metrics: {'rows': 140, 'open_mape_pct': 0.8837070954455782, 'high_mape_pct': 1.3736506877899224, 'low_mape_pct': 1.2691319195198223, 'close_mape_pct': 1.5920197712605746, 'close_direction_accuracy_pct': 51.42857142857142, 'status': 'SCORED'}

## Trading evidence
- Promotion gate: NOT_ELIGIBLE_INSUFFICIENT_OR_UNFAVORABLE_EVIDENCE
- Matched sessions: 33; V2 executed trades: 2

## Data and news audits
- Data checks: 6
- News checks: 6

Note: saved-prediction replay is not a model-refit walk-forward backtest. These reports do not change strategy decisions.
