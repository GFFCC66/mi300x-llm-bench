"""Small numeric helpers with no third-party dependency."""

from __future__ import annotations


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching ``numpy.percentile`` default.

    ``pct`` is in ``[0, 100]``. The rank is ``pct / 100 * (n - 1)`` on the
    sorted sample. With few requests, p99 sits near the maximum and is not a
    stable tail estimate.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if pct < 0 or pct > 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1 or pct == 0:
        return float(ordered[0])
    if pct == 100:
        return float(ordered[-1])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)
