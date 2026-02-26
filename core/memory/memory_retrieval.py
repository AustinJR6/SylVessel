from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List

from .memory_types import RetrievalQuery, RetrievalResult


_EMOTION_TYPE_WEIGHT = {
    "core_identity": 1.05,
    "relationship": 1.2,
    "belief": 1.1,
    "emotional": 1.2,
    "operational": 1.0,
    "dream": 1.0,
    "autobiographical": 1.1,
    "relational": 1.2,
    "contextual": 1.0,
}


class MemoryRetrievalPipeline:
    """Retrieval pipeline: type filter -> semantic search -> recency/emotion rerank."""

    def __init__(self, manager):
        self.manager = manager

    def _recency_boost(self, timestamp: Any, half_life_days: float = 14.0) -> float:
        try:
            ts = float(timestamp)
            age_days = max(0.0, (datetime.now().timestamp() - ts) / 86400.0)
            return math.exp(-age_days / max(1.0, half_life_days))
        except Exception:
            return 0.25

    def _score(self, item: Dict[str, Any]) -> float:
        similarity = float(item.get("similarity", 0.0))
        recency = self._recency_boost(item.get("timestamp"))
        emotion_weight = float(item.get("feeling_weight", 0.5))
        type_weight = _EMOTION_TYPE_WEIGHT.get(str(item.get("memory_type", "contextual")), 1.0)
        significance = float(item.get("significance_score", 0.5))
        return (0.50 * similarity) + (0.20 * recency) + (0.20 * emotion_weight) + (0.10 * significance * type_weight)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        type_filter = {m.value for m in (query.memory_types or [])}

        rows = self.manager.retrieve_memories(
            query.text,
            personality=query.identity,
            limit=max(query.limit * 3, 24),
            match_threshold=query.min_similarity,
        )
        if type_filter:
            rows = [r for r in rows if str(r.get("memory_type", "")).lower() in type_filter]

        for row in rows:
            row["pipeline_score"] = round(self._score(row), 6)

        rows.sort(key=lambda r: r.get("pipeline_score", 0.0), reverse=True)
        final = rows[: query.limit]
        core = self.manager.search_core_memories(query.text, k=2)
        return RetrievalResult(
            conversations=final,
            core_memories=core,
            scores={"candidates": len(rows), "selected": len(final)},
        )
