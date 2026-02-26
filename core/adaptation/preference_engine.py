from __future__ import annotations

from typing import Any, Dict


class PreferenceEngine:
    """Simple adaptive preference extraction from user feedback/events."""

    def __init__(self):
        self._prefs: Dict[str, Dict[str, Any]] = {}

    def update(self, identity: str, *, tone: str = "", verbosity: str = "", likes: str = "") -> None:
        slot = self._prefs.setdefault(identity, {})
        if tone:
            slot["tone"] = tone
        if verbosity:
            slot["verbosity"] = verbosity
        if likes:
            slot.setdefault("likes", [])
            if likes not in slot["likes"]:
                slot["likes"].append(likes)

    def get(self, identity: str) -> Dict[str, Any]:
        return dict(self._prefs.get(identity, {}))
