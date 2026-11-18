"""Pure date-to-partition helpers for historical research inputs."""

from datetime import date, datetime


def year_quarter(value: date | datetime) -> tuple[int, int]:
    """Return the calendar year and quarter for a date-like value."""
    if isinstance(value, datetime):
        value = value.date()
    if not isinstance(value, date):
        raise TypeError("value must be datetime.date or datetime.datetime")
    return value.year, ((value.month - 1) // 3) + 1


def partition_key(value: date | datetime) -> str:
    """Return a stable Hive-style year/quarter partition key."""
    year, quarter = year_quarter(value)
    return f"year={year:04d}/quarter=Q{quarter}"
