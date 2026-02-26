from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config_loader import config
from memory.memory_manager import MemoryManager
from memory.semantic_search import SemanticMemoryEngine
from memory.supabase_client import pooled_cursor

from .memory_interface import MemoryInterface
from .memory_types import MemoryRecord, MemoryType, RetrievalQuery, RetrievalResult
from .memory_retrieval import MemoryRetrievalPipeline
from .memory_consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)


_TYPE_MAP = {
    MemoryType.CORE_IDENTITY: "autobiographical",
    MemoryType.RELATIONSHIP: "relational",
    MemoryType.BELIEF: "autobiographical",
    MemoryType.EMOTIONAL: "emotional",
    MemoryType.OPERATIONAL: "contextual",
    MemoryType.DREAM: "contextual",
}


class SupabaseMemoryRepository(MemoryInterface):
    """Memory abstraction backed by existing Supabase + pgvector schema."""

    def __init__(self):
        self.manager = MemoryManager()
        self.semantic = SemanticMemoryEngine()
        self.retrieval = MemoryRetrievalPipeline(self.manager)
        self.consolidator = MemoryConsolidator(self.semantic)
        self.lazy_min_chars = int(getattr(config, "LAZY_EMBED_MIN_CHARS", 120))
        self.lazy_emotion_threshold = float(getattr(config, "LAZY_EMBED_EMOTION_THRESHOLD", 0.66))
        raw = (getattr(config, "ALLOWED_IDENTITIES", "sylana,claude") or "sylana,claude").strip()
        self.allowed_identities = {x.strip().lower() for x in raw.split(",") if x.strip()}
        self._ensure_identity_indexes()

    def _assert_identity(self, identity: str) -> str:
        normalized = (identity or "sylana").strip().lower()
        if normalized not in self.allowed_identities:
            raise ValueError(f"Identity '{identity}' is not allowed")
        return normalized

    def _ensure_identity_indexes(self) -> None:
        """Create identity-specific vector and lookup indexes when possible."""
        for ident in sorted(self.allowed_identities):
            safe_ident = "".join(ch for ch in ident if ch.isalnum() or ch == "_")
            try:
                with pooled_cursor(commit=True) as cur:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_memories_personality_{safe_ident}
                        ON memories (personality, timestamp DESC)
                        """
                    )
                    # Partial vector index by identity to reduce cross-identity scan cost.
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_memories_embedding_{safe_ident}
                        ON memories USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        WHERE embedding IS NOT NULL AND COALESCE(personality, 'sylana') = '{ident.replace("'", "''")}'
                        """
                    )
            except Exception as e:
                logger.debug("Identity index ensure skipped for %s: %s", ident, e)

    def _should_embed(self, memory: MemoryRecord) -> bool:
        if memory.important:
            return True
        text_len = len((memory.user_input or "") + (memory.sylana_response or ""))
        if text_len >= self.lazy_min_chars:
            return True
        score = abs(float(memory.emotion.valence)) + float(memory.emotion.arousal)
        return score >= self.lazy_emotion_threshold

    def _store_without_embedding(self, memory: MemoryRecord) -> int:
        identity = self._assert_identity(memory.identity)
        emotion_data = {
            "category": memory.emotion.category,
            "emotion": memory.emotion.category,
            "intensity": memory.emotion.intensity,
            "vad": {
                "valence": memory.emotion.valence,
                "arousal": memory.emotion.arousal,
                "dominance": memory.emotion.dominance,
            },
        }
        metadata = memory.metadata or {}
        memory_type = _TYPE_MAP.get(memory.memory_type, "contextual")
        now_ts = memory.timestamp if memory.timestamp is not None else datetime.now().timestamp()
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    user_input, sylana_response, timestamp, emotion, embedding, personality, privacy_level,
                    thread_id, memory_type, feeling_weight, energy_shift, comfort_level, significance_score, secure_payload
                ) VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    memory.user_input,
                    memory.sylana_response,
                    now_ts,
                    memory.emotion.category,
                    identity,
                    metadata.get("privacy_level", "private"),
                    metadata.get("thread_id"),
                    memory_type,
                    float(metadata.get("feeling_weight", 0.5)),
                    float(metadata.get("energy_shift", 0.0)),
                    float(metadata.get("comfort_level", 0.5)),
                    float(metadata.get("significance_score", 0.5)),
                    json.dumps({"emotion_data": emotion_data, "metadata": metadata}),
                ),
            )
            memory_id = int(cur.fetchone()[0])
            return memory_id

    def store(self, memory: MemoryRecord) -> int:
        identity = self._assert_identity(memory.identity)
        emotion_data = {
            "emotion": memory.emotion.category,
            "category": memory.emotion.category,
            "intensity": memory.emotion.intensity,
            "valence": memory.emotion.valence,
            "arousal": memory.emotion.arousal,
            "dominance": memory.emotion.dominance,
        }
        if self._should_embed(memory):
            return self.manager.store_conversation(
                user_input=memory.user_input,
                sylana_response=memory.sylana_response,
                emotion=memory.emotion.category,
                emotion_data=emotion_data,
                personality=identity,
                privacy_level=(memory.metadata or {}).get("privacy_level", "private"),
                thread_id=(memory.metadata or {}).get("thread_id"),
            )
        logger.debug("Storing memory without embedding due to lazy-embed threshold")
        return self._store_without_embedding(memory)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        query.identity = self._assert_identity(query.identity)
        return self.retrieval.retrieve(query)

    def history(self, identity: str, limit: int = 5) -> List[Dict[str, Any]]:
        scoped = self._assert_identity(identity)
        return self.manager.get_conversation_history(limit=limit, personality=scoped)

    def stats(self) -> Dict[str, Any]:
        return self.manager.get_stats()

    def consolidate(self, identity: Optional[str] = None, archive: bool = True) -> Dict[str, Any]:
        return self.consolidator.consolidate(identity=identity, archive=archive)

    def close(self) -> None:
        self.manager.close()

    async def store_async(self, memory: MemoryRecord) -> int:
        return await asyncio.to_thread(self.store, memory)

    async def retrieve_async(self, query: RetrievalQuery) -> RetrievalResult:
        return await asyncio.to_thread(self.retrieve, query)

    async def history_async(self, identity: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.history, identity, limit)

