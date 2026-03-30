"""Route text to the correct sentiment model."""
from __future__ import annotations

from typing import Any


def route(text: str, lang: str = "en") -> Any:
    """Return placeholder model based on language or content."""
    if lang != "en":
        return "multilingual-model"
    return "finbert"

