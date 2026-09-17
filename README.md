# Stock Picker

A GitHub-native research pipeline for Nifty Midcap 150 stocks.

## Storage

GitHub is the primary persistent store. The workflow writes partitioned CSV/JSON files into `data/` and trained model metadata into `models/`. GitHub Actions runs the scheduled jobs and commits updated data back to the repository.

The project deliberately avoids a separate database for the free version.

## Pipeline

1. Download/update Nifty Midcap 150 daily OHLCV data.
2. Calculate technical indicators.
3. Collect available fundamental metrics.
4. Score and rank the 150 stocks.
5. Select the top 10.
6. Before market open, predict next-session Open/High/Low/Close.
7. Store the predictions in GitHub.
8. After market close, download actual OHLC.
9. Compare actual vs predicted values.
10. Append the observations to the training dataset and retrain periodically.

## Important

This is a research/forecasting project, not financial advice. Yahoo Finance data availability and licensing should be checked before relying on it for production use.
