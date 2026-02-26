from __future__ import annotations

"""
Compatibility server entrypoint.
Keeps FastAPI app behavior from root server.py while instantiating unified Brain.
"""

from core.brain import Brain
from server import app  # noqa: F401

brain = Brain.create_default(mode="claude")

__all__ = ["app", "brain"]
