"""Macro aware sentiment adjustments."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from typing import Dict


def load_macro_calendar(path: str | Path = "data/macro_event_calendar.json") -> Dict:
    """Load macro event calendar from JSON file."""
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def adjust_for_macro(score: float, calendar: Dict, date: datetime) -> float:
    """Reduce score around major macro events."""
    events = calendar.get(date.strftime("%Y-%m-%d"), [])
    if events:
        return score * 0.8
    return score

