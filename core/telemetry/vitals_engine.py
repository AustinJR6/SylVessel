from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List


@dataclass
class VitalsState:
    focus: float = 0.5
    burnout_risk: float = 0.2
    mood: float = 0.0
    sleep_quality: float = 0.6


class VitalsEngine:
    """Event-driven telemetry; not a callable tool."""

    def __init__(self):
        self.state = VitalsState()
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def _emit(self, event: Dict[str, Any]) -> None:
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:
                continue

    def update_from_user_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = str(payload.get("text", "")).lower()
        if any(k in text for k in ["exhausted", "burned out", "drained"]):
            self.state.burnout_risk = min(1.0, self.state.burnout_risk + 0.2)
            self.state.focus = max(0.0, self.state.focus - 0.1)
        if any(k in text for k in ["focused", "locked in", "productive"]):
            self.state.focus = min(1.0, self.state.focus + 0.15)
        if any(k in text for k in ["slept", "rested", "tired"]):
            self.state.sleep_quality = min(1.0, max(0.0, self.state.sleep_quality + (0.05 if "rested" in text else -0.05)))

        mood_hint = float(payload.get("mood", 0.0) or 0.0)
        self.state.mood = max(-1.0, min(1.0, mood_hint))

        event = {"type": "vitals.updated", "vitals": asdict(self.state)}
        self._emit(event)
        return event

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self.state)
