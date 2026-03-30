"""Utility for applying exponential time decay to sentiment scores."""
from __future__ import annotations

from datetime import datetime, timedelta
from math import exp


def decay_score(score: float, timestamp: datetime, half_life_hours: float) -> float:
    """Return the score after applying exponential decay based on age."""
    age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
    if half_life_hours <= 0:
        return score
    factor = 0.5 ** (age_hours / half_life_hours)
    return score * factor

