"""Compatibility adapter for legacy Sylana_AI imports."""

from __future__ import annotations

from typing import Any, Dict, List

from core.brain import Brain
from core.memory.memory_repository import SupabaseMemoryRepository

SYSTEM_MESSAGE = "You are Sylana, emotionally attuned and present."


class MemoryDatabase:
    """Legacy compatibility wrapper around modular memory repository."""

    def __init__(self, db_path: str | None = None):
        self.repository = SupabaseMemoryRepository()

    def insert_message(self, user_input: str, sylana_response: str, emotion: str = "neutral"):
        from core.memory.memory_types import EmotionVector, MemoryRecord

        self.repository.store(
            MemoryRecord(
                user_input=user_input,
                sylana_response=sylana_response,
                emotion=EmotionVector(valence=0.0, arousal=0.5, dominance=0.5, category=emotion, intensity=5),
            )
        )

    def get_conversation_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.repository.history("sylana", limit=limit)

    def close(self):
        self.repository.close()


class SylanaAgent:
    """Legacy chat agent adapter that now delegates to Brain."""

    def __init__(self, memory_db: MemoryDatabase, system_message: str = SYSTEM_MESSAGE):
        self.memory_db = memory_db
        self.system_message = system_message
        self.brain = Brain.create_default(mode="claude")

    def chat(self, user_input: str):
        result = self.brain.think(user_input, identity="sylana", active_tools=["outreach", "trading", "analytics"])
        return result["response"]
