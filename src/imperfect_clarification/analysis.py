"""Reusable calculations for the UserBench exploratory analysis."""

from collections.abc import Mapping, Sequence
from fractions import Fraction


def count_field_presence(
    records: Sequence[Mapping[str, object]], key: str
) -> dict[str, int]:
    """Count absent keys, explicit nulls and values equal to zero."""
    return {
        "absent": sum(key not in record for record in records),
        "explicit_null": sum(key in record and record[key] is None for record in records),
        "numeric_zero": sum(key in record and record[key] == 0 for record in records),
    }


def diagnostic_percentages(metrics: Mapping[str, str | int | float]) -> dict[str, float]:
    """Convert the diagnostic's fractional metrics to percentages."""
    return {name: 100 * float(Fraction(value)) for name, value in metrics.items()}
