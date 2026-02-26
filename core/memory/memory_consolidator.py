from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from memory.supabase_client import pooled_cursor

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Compresses older memories into meta-memory summaries."""

    def __init__(self, semantic_engine):
        self.semantic_engine = semantic_engine

    def _cluster_key(self, row: Dict[str, Any]) -> str:
        ts = row.get("timestamp")
        month = "unknown"
        try:
            month = datetime.fromtimestamp(float(ts)).strftime("%Y-%m")
        except Exception:
            pass
        return f"{row.get('personality','sylana')}|{row.get('memory_type','contextual')}|{month}"

    def consolidate(self, identity: Optional[str] = None, archive: bool = True) -> Dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=45)
        params: List[Any] = [cutoff.timestamp()]
        where = "timestamp < %s"
        if identity:
            where += " AND COALESCE(personality, 'sylana') = %s"
            params.append(identity)

        try:
            with pooled_cursor(commit=False) as cur:
                cur.execute(
                    f"""
                    SELECT id, user_input, sylana_response, timestamp, COALESCE(personality, 'sylana') AS personality,
                           COALESCE(memory_type, 'contextual') AS memory_type
                    FROM memories
                    WHERE {where}
                    ORDER BY timestamp ASC
                    LIMIT 1200
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        except Exception as e:
            logger.error("Consolidation scan failed: %s", e)
            return {"status": "error", "error": str(e)}

        bucketed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            item = {
                "id": r[0],
                "user_input": r[1] or "",
                "sylana_response": r[2] or "",
                "timestamp": r[3],
                "personality": r[4],
                "memory_type": r[5],
            }
            bucketed[self._cluster_key(item)].append(item)

        created = 0
        removed = 0
        archived_rows = 0
        with pooled_cursor(commit=True) as cur:
            for key, items in bucketed.items():
                if len(items) < 6:
                    continue
                sample = items[:8]
                summary = " | ".join(
                    f"U:{x['user_input'][:70]} A:{x['sylana_response'][:70]}" for x in sample
                )
                persona = items[0]["personality"]
                mtype = items[0]["memory_type"]
                text = f"[META-MEMORY {key}] {summary}"
                try:
                    emb = self.semantic_engine.encode_text(text)
                    cur.execute(
                        """
                        INSERT INTO memories (
                            user_input, sylana_response, timestamp, emotion, embedding, personality, privacy_level,
                            memory_type, significance_score
                        ) VALUES (%s, %s, %s, %s, %s, %s, 'private', %s, %s)
                        """,
                        (
                            f"Consolidated memory cluster ({key})",
                            summary,
                            datetime.now().timestamp(),
                            "neutral",
                            emb,
                            persona,
                            mtype,
                            0.95,
                        ),
                    )
                    created += 1

                    ids = [x["id"] for x in items]
                    if archive:
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS memory_archive (
                                archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                memory_id BIGINT NOT NULL,
                                payload JSONB NOT NULL
                            )
                            """
                        )
                        for x in items:
                            cur.execute(
                                "INSERT INTO memory_archive (memory_id, payload) VALUES (%s, %s::jsonb)",
                                (x["id"], '{"kind":"consolidated"}'),
                            )
                            archived_rows += 1

                    cur.execute("DELETE FROM memories WHERE id = ANY(%s)", (ids,))
                    removed += len(ids)
                except Exception as e:
                    logger.warning("Consolidation cluster failed (%s): %s", key, e)
                    continue

        return {
            "status": "success",
            "clusters_created": created,
            "source_rows_removed": removed,
            "archived_rows": archived_rows,
        }
