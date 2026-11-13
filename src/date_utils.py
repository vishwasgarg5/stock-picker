from __future__ import annotations

import pandas as pd


def parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Parse date strings with mixed date-only and timestamp representations.

    Uses pandas' mixed-format parser when available and a per-value fallback for
    older pandas releases. Invalid values remain NaT; callers decide how to
    handle them. The returned values are timezone-naive timestamps.
    """
    source = values.copy()
    try:
        parsed = pd.to_datetime(source, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        parsed = source.map(
            lambda value: pd.to_datetime(value, errors="coerce", utc=True)
        )
    return parsed.dt.tz_convert(None)


def normalize_date_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy with a consistently parsed, midnight-normalized date column."""
    result = frame.copy()
    if column in result.columns:
        result[column] = parse_mixed_dates(result[column]).dt.normalize()
    return result
