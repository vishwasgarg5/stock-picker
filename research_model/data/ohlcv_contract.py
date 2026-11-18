"""Input schema constants for the isolated research data layer."""

REQUIRED_OHLCV_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

OPTIONAL_CORPORATE_ACTION_COLUMNS = (
    "dividends",
    "stock_splits",
)

SOURCE_DATASET = "data/ohlcv.csv"
