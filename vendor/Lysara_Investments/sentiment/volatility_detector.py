"""Detect sentiment volatility across data sources."""
from __future__ import annotations

from typing import Dict, List
import statistics


def compute_volatility(sentiments: Dict[str, float]) -> float:
    """Return standard deviation of sentiment scores as a simple volatility index."""
    values: List[float] = list(sentiments.values())
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)

