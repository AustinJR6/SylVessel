"""
Sylana Vessel - Unified Memory Manager
Central interface for all memory operations with Supabase + pgvector
"""

import logging
import json
import math
import re
from typing import List, Dict, Optional, Any
from datetime import datetime

from core.config_loader import config
from memory.semantic_search import SemanticMemoryEngine
from memory.supabase_client import get_connection, close_connection

logger = logging.getLogger(__name__)


# Emotion weights for importance scoring
EMOTION_WEIGHTS = {
    "ecstatic": 2.0,
    "devastated": 2.0,
    "happy": 1.5,
    "sad": 1.5,
    "neutral": 1.0
}

MEMORY_TYPE_WEIGHTS = {
    "autobiographical": 1.2,
    "relational": 1.35,
    "contextual": 1.0,
    "emotional": 1.25,
}

QUERY_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "you", "your",
    "are", "was", "were", "about", "what", "when", "where", "why", "how", "can",
    "did", "our", "we", "she", "him", "her", "they", "them", "into", "onto",
}


class MemoryManager:
    """
    Unified memory management system combining:
    - Supabase PostgreSQL storage
    - pgvector semantic search
    - Core memory retrieval
    - Importance scoring
    """

    def __init__(self, db_path=None):
        self.semantic_engine = None
        self.db_path = db_path  # Backward-compat only (Supabase is authoritative).

        # Initialize components
        self._verify_connection()
        self._initialize_semantic_engine()

        logger.info("MemoryManager initialized with Supabase backend")

    def _verify_connection(self):
        """Verify Supabase connection works"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            logger.info("Database connection established")
        except Exception as e:
            logger.exception(f"Failed to connect to Supabase: {e}")
            raise

    def _initialize_semantic_engine(self):
        """Initialize semantic search engine"""
        self.semantic_engine = SemanticMemoryEngine()

    def rebuild_index(self):
        """No-op: pgvector handles indexing automatically."""
        logger.info("Index rebuild requested — pgvector handles this automatically")

    def _classify_memory_type(self, user_input: str, response: str, emotion_data: Optional[Dict[str, Any]]) -> str:
        text = f"{(user_input or '').lower()} {(response or '').lower()}"
        emotion = (emotion_data or {}).get("category", "")
        if any(k in text for k in ["i feel", "i'm feeling", "this felt", "energy", "comfort", "safe with you"]):
            return "emotional"
        if any(k in text for k in ["we", "us", "our", "relationship", "between us", "bond"]):
            return "relational"
        if any(k in text for k in ["i remember", "i learned", "i changed", "growth", "who i am"]):
            return "autobiographical"
        if emotion in {"devastated", "sad", "happy", "ecstatic"}:
            return "emotional"
        return "contextual"

    def _compute_feeling_weight(self, emotion_data: Optional[Dict[str, Any]]) -> float:
        if not emotion_data:
            return 0.5
        category = (emotion_data.get("category") or "neutral").lower()
        intensity = float(emotion_data.get("intensity", 5))
        base = EMOTION_WEIGHTS.get(category, 1.0)
        normalized_intensity = max(0.1, min(1.0, intensity / 10.0))
        return round(max(0.1, min(2.5, base * normalized_intensity)), 3)

    def _compute_energy_shift(self, emotion_data: Optional[Dict[str, Any]]) -> float:
        if not emotion_data:
            return 0.0
        category = (emotion_data.get("category") or "neutral").lower()
        mapping = {
            "ecstatic": 0.9, "happy": 0.5, "neutral": 0.0, "sad": -0.5, "devastated": -0.9,
            "anxious": -0.4, "frustrated": -0.45, "curious": 0.2, "longing": -0.2,
        }
        return round(mapping.get(category, 0.0), 3)

    def _compute_comfort_level(self, user_input: str, response: str, emotion_data: Optional[Dict[str, Any]]) -> float:
        text = f"{(user_input or '').lower()} {(response or '').lower()}"
        boost = 0.0
        if any(k in text for k in ["safe", "comfort", "with you", "trust", "supported", "heard"]):
            boost += 0.2
        if any(k in text for k in ["alone", "afraid", "panic", "overwhelmed"]):
            boost -= 0.15
        category = (emotion_data or {}).get("category", "neutral")
        if category in {"happy", "ecstatic"}:
            boost += 0.15
        elif category in {"devastated", "sad"}:
            boost -= 0.1
        return round(max(0.0, min(1.0, 0.5 + boost)), 3)

    def _compute_significance_score(self, memory_type: str, feeling_weight: float, comfort_level: float) -> float:
        type_weight = MEMORY_TYPE_WEIGHTS.get(memory_type, 1.0)
        score = (0.55 * feeling_weight) + (0.25 * type_weight) + (0.2 * comfort_level)
        return round(max(0.05, min(2.5, score)), 3)

    def _extract_query_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9']+", (query or "").lower())
        out = []
        for tok in tokens:
            if len(tok) < 3 or tok in QUERY_STOPWORDS:
                continue
            out.append(tok)
        seen = set()
        deduped = []
        for tok in out:
            if tok in seen:
                continue
            seen.add(tok)
            deduped.append(tok)
        return deduped[:10]

    def _fts_content_search(self, query: str, personality: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Full-text search using Postgres tsvector/plainto_tsquery.
        Understands stemming, plurals, stop words — far smarter than ILIKE.
        Returns empty list if fts_vector column is not yet present.
        """
        query_text = (query or "").strip()
        if not query_text:
            return []

        conn = get_connection()
        cur = conn.cursor()
        fetch_limit = max(limit * 3, 60)
        try:
            cur.execute(
                """
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       COALESCE(personality, 'sylana') AS personality,
                       COALESCE(memory_type, 'contextual') AS memory_type,
                       COALESCE(feeling_weight, 0.5) AS feeling_weight,
                       COALESCE(significance_score, 0.5) AS significance_score,
                       ts_rank(fts_vector, plainto_tsquery('english', %s)) AS fts_rank
                FROM memories
                WHERE fts_vector @@ plainto_tsquery('english', %s)
                  AND (%s IS NULL OR COALESCE(personality, 'sylana') = %s)
                ORDER BY fts_rank DESC, significance_score DESC, id DESC
                LIMIT %s
                """,
                (query_text, query_text, personality, personality, fetch_limit),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"FTS search unavailable (column may not exist yet): {e}")
            return []

        out: List[Dict[str, Any]] = []
        for row in rows:
            fts_rank = float(row[9] or 0.0)
            out.append({
                "id": row[0],
                "user_input": row[1] or "",
                "sylana_response": row[2] or "",
                "emotion": row[3] or "neutral",
                "timestamp": row[4],
                "personality": row[5] or "sylana",
                "memory_type": row[6] or "contextual",
                "feeling_weight": float(row[7] or 0.5),
                "significance_score": float(row[8] or 0.5),
                "keyword_score": fts_rank,
                "similarity": fts_rank,
                "text": f"User: {row[1]}\nSylana: {row[2]}",
            })
        return out[:max(limit, 20)]

    def _keyword_content_search(self, query: str, personality: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Content search over raw memory text.
        Tries Postgres full-text search (tsvector) first for semantic precision;
        falls back to ILIKE token overlap when FTS index is not yet available.
        """
        # --- Primary: full-text search (stemming-aware, no stop words) ---
        fts_results = self._fts_content_search(query, personality, limit)
        if fts_results:
            return fts_results

        # --- Fallback: ILIKE keyword overlap ---
        keywords = self._extract_query_keywords(query)
        if not keywords:
            return []

        conn = get_connection()
        cur = conn.cursor()
        rows: List[Any] = []
        like_clauses = []
        params: List[Any] = [personality, personality]
        for kw in keywords[:8]:
            like = f"%{kw}%"
            like_clauses.append("(user_input ILIKE %s OR sylana_response ILIKE %s)")
            params.extend([like, like])
        where = " OR ".join(like_clauses) if like_clauses else "FALSE"
        try:
            cur.execute(
                f"""
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       COALESCE(personality, 'sylana') AS personality,
                       COALESCE(memory_type, 'contextual') AS memory_type,
                       COALESCE(feeling_weight, 0.5) AS feeling_weight,
                       COALESCE(significance_score, 0.5) AS significance_score
                FROM memories
                WHERE (%s IS NULL OR COALESCE(personality, 'sylana') = %s)
                  AND ({where})
                ORDER BY significance_score DESC, id DESC
                LIMIT %s
                """,
                [*params, max(limit * 20, 160)],
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.warning(f"Keyword content search failed: {e}")
            rows = []

        out: List[Dict[str, Any]] = []
        phrase = (query or "").lower().strip()
        for row in rows:
            user_text = row[1] or ""
            assistant_text = row[2] or ""
            merged = f"{user_text}\n{assistant_text}".lower()
            unique_hits = sum(1 for kw in keywords if kw in merged)
            total_hits = sum(merged.count(kw) for kw in keywords)
            phrase_bonus = 1.0 if phrase and phrase in merged else 0.0
            keyword_score = (
                (1.4 * phrase_bonus) +
                (0.8 * (unique_hits / max(1, len(keywords)))) +
                (0.08 * min(total_hits, 8))
            )
            out.append({
                "id": row[0],
                "user_input": user_text,
                "sylana_response": assistant_text,
                "emotion": row[3] or "neutral",
                "timestamp": row[4],
                "personality": row[5] or "sylana",
                "memory_type": row[6] or "contextual",
                "feeling_weight": float(row[7] or 0.5),
                "significance_score": float(row[8] or 0.5),
                "keyword_score": float(keyword_score),
                "similarity": float(keyword_score),
                "text": f"User: {row[1]}\nSylana: {row[2]}",
            })
        out.sort(key=lambda x: (x.get("keyword_score", 0.0), x.get("significance_score", 0.0), x.get("id", 0)), reverse=True)
        return out[:max(limit, 20)]

    def _encrypt_payload(self, payload: Dict[str, Any]) -> Optional[bytes]:
        key = getattr(config, "MEMORY_ENCRYPTION_KEY", None)
        if not key:
            return None
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT pgp_sym_encrypt(%s, %s)", (json.dumps(payload), key))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.warning(f"Secure payload encryption failed: {e}")
            return None

    def _decrypt_payload(self, encrypted_payload: Any) -> Dict[str, Any]:
        key = getattr(config, "MEMORY_ENCRYPTION_KEY", None)
        if not key or encrypted_payload is None:
            return {}
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT pgp_sym_decrypt(%s::bytea, %s)", (encrypted_payload, key))
            row = cur.fetchone()
            if not row or not row[0]:
                return {}
            return json.loads(row[0])
        except Exception:
            return {}

    def store_conversation(
        self,
        user_input: str,
        sylana_response: str,
        emotion: str = "neutral",
        emotion_data: Optional[Dict[str, Any]] = None,
        personality: str = "sylana",
        privacy_level: str = "private",
        thread_id: int = None,
    ) -> int:
        """
        Store a conversation turn with embedding for vector search.

        Returns:
            ID of inserted memory
        """
        conn = get_connection()
        cur = conn.cursor()

        # Generate embedding at insert time
        text = f"User: {user_input}\nSylana: {sylana_response}"
        embedding = self.semantic_engine.encode_text(text)

        memory_type = self._classify_memory_type(user_input, sylana_response, emotion_data)
        feeling_weight = self._compute_feeling_weight(emotion_data)
        energy_shift = self._compute_energy_shift(emotion_data)
        comfort_level = self._compute_comfort_level(user_input, sylana_response, emotion_data)
        significance_score = self._compute_significance_score(memory_type, feeling_weight, comfort_level)
        secure_payload = self._encrypt_payload({
            "user_input": user_input,
            "sylana_response": sylana_response,
            "emotion_data": emotion_data or {"category": emotion},
            "thread_id": thread_id,
            "personality": personality,
            "memory_type": memory_type,
            "stored_at": datetime.utcnow().isoformat(),
        })

        try:
            cur.execute("""
                INSERT INTO memories
                (user_input, sylana_response, timestamp, emotion, embedding, personality, privacy_level, thread_id,
                 memory_type, feeling_weight, energy_shift, comfort_level, significance_score, secure_payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                user_input,
                sylana_response,
                datetime.now().timestamp(),
                emotion,
                embedding,
                personality,
                privacy_level,
                thread_id,
                memory_type,
                feeling_weight,
                energy_shift,
                comfort_level,
                significance_score,
                secure_payload,
            ))
            memory_id = cur.fetchone()[0]
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store conversation: {e}")
            raise

        self._update_session_continuity_state(
            personality=personality,
            memory_type=memory_type,
            emotion_data=emotion_data or {"category": emotion, "intensity": 5},
            feeling_weight=feeling_weight,
            user_input=user_input,
            sylana_response=sylana_response,
        )
        logger.info(f"Stored conversation {memory_id} with emotion: {emotion}")
        return memory_id

    def store_message(
        self,
        message: str,
        response: str,
        personality: str,
        thread_id: int = None,
        privacy_level: str = "private",
        emotion: str = "neutral",
    ) -> int:
        """Compatibility helper for personality-aware chat flows."""
        return self.store_conversation(
            user_input=message,
            sylana_response=response,
            emotion=emotion,
            personality=personality,
            privacy_level=privacy_level,
            thread_id=thread_id,
        )

    def _apply_continuity_ranking(self, conversations: List[Dict]) -> List[Dict]:
        if not conversations:
            return []
        half_life_days = max(1, int(getattr(config, "MEMORY_DECAY_HALF_LIFE_DAYS", 14)))
        now_ts = datetime.now().timestamp()
        for conv in conversations:
            similarity = float(conv.get("similarity", 0.0))
            keyword_score = float(conv.get("keyword_score", 0.0))
            significance = float(conv.get("significance_score") or conv.get("feeling_weight") or 0.5)
            ts = conv.get("timestamp")
            try:
                age_days = max(0.0, (now_ts - float(ts)) / 86400.0) if ts is not None else float(half_life_days)
            except Exception:
                age_days = float(half_life_days)
            recency = math.exp(-math.log(2.0) * (age_days / half_life_days))
            # Content relevance has priority for legacy imported memories with noisy dates.
            conv["continuity_score"] = round(
                (0.45 * similarity) + (0.2 * significance) + (0.3 * keyword_score) + (0.05 * recency),
                4,
            )
        conversations.sort(key=lambda x: x.get("continuity_score", 0.0), reverse=True)
        return conversations

    def _touch_memories(self, memory_ids: List[int]) -> None:
        if not memory_ids:
            return
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE memories
                SET access_count = COALESCE(access_count, 0) + 1,
                    last_accessed_at = NOW()
                WHERE id = ANY(%s)
                """,
                (memory_ids,),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Failed to update memory access metadata: {e}")

    def load_startup_continuity(self) -> Dict[str, Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        snapshots: Dict[str, Dict[str, Any]] = {}
        try:
            cur.execute("SELECT personality, encrypted_state FROM session_continuity_state")
            for personality, encrypted_state in cur.fetchall():
                snapshots[personality] = self._decrypt_payload(encrypted_state)
        except Exception as e:
            logger.warning(f"Failed loading startup continuity state: {e}")
        return snapshots

    def get_session_continuity(self, personality: str = "sylana") -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        continuity: Dict[str, Any] = {}
        try:
            cur.execute("SELECT encrypted_state FROM session_continuity_state WHERE personality = %s", (personality,))
            row = cur.fetchone()
            if row and row[0]:
                continuity = self._decrypt_payload(row[0])
        except Exception:
            continuity = {}

        try:
            lookback_days = max(1, int(getattr(config, "CONTINUITY_LOOKBACK_DAYS", 30)))
            limit = max(3, int(getattr(config, "MAX_CONTINUITY_ITEMS", 8)))
            cur.execute(
                """
                SELECT user_input, sylana_response, memory_type, emotion, feeling_weight, significance_score, timestamp
                FROM memories
                WHERE personality = %s
                  AND timestamp >= (EXTRACT(EPOCH FROM NOW()) - (%s * 86400))
                ORDER BY significance_score DESC NULLS LAST, timestamp DESC
                LIMIT %s
                """,
                (personality, lookback_days, limit),
            )
            continuity["recent_weighted_memories"] = [{
                "user_input": r[0] or "",
                "sylana_response": r[1] or "",
                "memory_type": r[2] or "contextual",
                "emotion": r[3] or "neutral",
                "feeling_weight": float(r[4] or 0.5),
                "significance_score": float(r[5] or 0.5),
                "timestamp": r[6],
            } for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to build continuity context: {e}")
            continuity.setdefault("recent_weighted_memories", [])
        return continuity

    def _update_session_continuity_state(
        self,
        personality: str,
        memory_type: str,
        emotion_data: Optional[Dict[str, Any]],
        feeling_weight: float,
        user_input: str,
        sylana_response: str,
    ) -> None:
        if not getattr(config, "MEMORY_ENCRYPTION_KEY", None):
            return

        text = f"{(user_input or '').lower()} {(sylana_response or '').lower()}"
        baseline = (emotion_data or {}).get("category", "neutral")
        momentum = "rising" if baseline in {"happy", "ecstatic", "curious"} else "steady"
        if baseline in {"sad", "devastated", "frustrated", "anxious"}:
            momentum = "fragile"

        state_payload = {
            "last_emotion": baseline,
            "emotional_baseline": baseline,
            "relationship_trust_level": round(max(0.0, min(1.0, 0.55 + (feeling_weight - 0.5) * 0.2)), 3),
            "conversation_momentum": momentum,
            "communication_patterns": [
                "reassurance-seeking" if "need" in text or "help" in text else "reflective",
                "emotionally-open" if any(w in text for w in ["feel", "heart", "love", "hurt"]) else "practical",
            ],
            "active_projects": [w for w in ["deploy", "app", "memory", "chat", "cloud run"] if w in text][:4],
            "preference_signals": [w for w in ["concise", "detail", "sources", "fast", "gentle"] if w in text][:4],
            "last_memory_type": memory_type,
            "updated_at": datetime.utcnow().isoformat(),
        }

        encrypted_state = self._encrypt_payload(state_payload)
        if encrypted_state is None:
            return

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO session_continuity_state (personality, encrypted_state, version, updated_at)
                VALUES (%s, %s, 1, NOW())
                ON CONFLICT (personality)
                DO UPDATE SET encrypted_state = EXCLUDED.encrypted_state,
                              updated_at = NOW(),
                              version = session_continuity_state.version + 1
                """,
                (personality, encrypted_state),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Failed to persist continuity state: {e}")

    def _fetch_imported_context(
        self, query: str, personality: str, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Dedicated search lane for ChatGPT-imported memories (conversation_id IS NOT NULL).

        Runs a direct pgvector similarity query scoped to imported rows so that
        recent live conversations cannot crowd them out of the candidate pool.
        Returns an empty list on any error (non-fatal).
        """
        conn = get_connection()
        cur = conn.cursor()
        try:
            query_vec = self.semantic_engine.encode_query(query)
            cur.execute(
                """
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       COALESCE(personality, 'sylana') AS personality,
                       COALESCE(memory_type, 'contextual') AS memory_type,
                       COALESCE(feeling_weight, 0.5) AS feeling_weight,
                       COALESCE(significance_score, 0.5) AS significance_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND conversation_id IS NOT NULL
                  AND COALESCE(personality, 'sylana') = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, personality, query_vec, limit),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"Import lane search skipped: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "user_input": row[1] or "",
                    "sylana_response": row[2] or "",
                    "emotion": row[3] or "neutral",
                    "timestamp": row[4],
                    "personality": row[5],
                    "memory_type": row[6],
                    "feeling_weight": float(row[7]),
                    "significance_score": float(row[8]),
                    "similarity": float(row[9] or 0.0),
                    "keyword_score": 0.0,
                    "text": f"User: {row[1]}\nSylana: {row[2]}",
                }
            )
        return results

    def retrieve_memories(self, query: str, personality: str, limit: int = 15, match_threshold: float = 0.25) -> List[Dict]:
        """
        Retrieve memories via personality-aware SQL function.
        Falls back to regular semantic search if function is unavailable.

        Always supplements results with a dedicated import lane so that
        ChatGPT-imported memories cannot be fully crowded out by recent
        live conversations.
        """
        conn = get_connection()
        cur = conn.cursor()
        query_vec = self.semantic_engine.encode_query(query)

        # Dedicated import lane — runs in parallel with main search.
        # Fetches top-N imported memories by semantic similarity, regardless
        # of their significance_score vs live conversations.
        import_lane_limit = max(3, limit // 3)
        import_hits = self._fetch_imported_context(query, personality, limit=import_lane_limit)

        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, personality, similarity, emotion, memory_timestamp
                FROM match_memories(%s::vector, %s, %s, %s)
            """, (query_vec, float(match_threshold), int(limit), personality))
            rows = cur.fetchall()
            out = []
            for row in rows:
                out.append({
                    "id": row[0],
                    "user_input": row[1] or "",
                    "sylana_response": row[2] or "",
                    "personality": row[3] or "sylana",
                    "similarity": float(row[4] or 0.0),
                    "emotion": row[5] or "",
                    "timestamp": row[6],
                    "text": f"User: {row[1]}\nSylana: {row[2]}",
                })
            keyword_hits = self._keyword_content_search(query, personality=personality, limit=max(limit, 12))
            merged = []
            seen = set()
            for item in keyword_hits + out + import_hits:
                mem_id = item.get("id")
                if mem_id in seen:
                    continue
                seen.add(mem_id)
                merged.append(item)

            self._enrich_conversations(merged)
            ranked = self._apply_continuity_ranking(merged)

            # Guarantee at least one imported memory slot if imported memories
            # exist in the candidate pool but were ranked below the cutoff.
            min_import_slots = max(1, limit // 4)
            imported_in_top = [m for m in ranked[:limit] if m.get("conversation_id")]
            if len(imported_in_top) < min_import_slots:
                # Pull the highest-ranked imported memories from the full pool
                extra_imports = [
                    m for m in ranked[limit:]
                    if m.get("conversation_id")
                ][:min_import_slots - len(imported_in_top)]
                non_imported = [m for m in ranked[:limit] if not m.get("conversation_id")]
                # Replace the lowest-ranked non-imported slots with extra imports
                trimmed_non_imported = non_imported[: limit - len(imported_in_top) - len(extra_imports)]
                final = (imported_in_top + extra_imports + trimmed_non_imported)[:limit]
            else:
                final = ranked[:limit]

            self._touch_memories([m.get("id") for m in final if m.get("id")])
            return final
        except Exception as e:
            logger.warning(f"match_memories function unavailable, using fallback search: {e}")
            fallback = self.semantic_engine.search(query, k=limit, similarity_threshold=match_threshold)
            filtered = []
            if fallback:
                conn = get_connection()
                cur = conn.cursor()
                ids = [m.get("id") for m in fallback if m.get("id")]
                persona_map = {}
                if ids:
                    try:
                        cur.execute(
                            "SELECT id, COALESCE(personality, 'sylana') FROM memories WHERE id = ANY(%s)",
                            (ids,),
                        )
                        persona_map = {r[0]: r[1] for r in cur.fetchall()}
                    except Exception:
                        persona_map = {}

                for mem in fallback:
                    mem_id = mem.get("id")
                    mem_persona = persona_map.get(mem_id, "sylana")
                    if mem_persona == personality:
                        mem["personality"] = mem_persona
                        filtered.append(mem)
            keyword_hits = self._keyword_content_search(query, personality=personality, limit=max(limit, 12))
            merged = []
            seen = set()
            for item in keyword_hits + filtered + import_hits:
                mem_id = item.get("id")
                if mem_id in seen:
                    continue
                seen.add(mem_id)
                merged.append(item)

            self._enrich_conversations(merged)
            ranked = self._apply_continuity_ranking(merged)[:limit]
            self._touch_memories([m.get("id") for m in ranked if m.get("id")])
            return ranked

    def recall_relevant(
        self,
        query: str,
        k: int = None,
        include_core: bool = True,
        use_recency_boost: bool = True,
        personality: str = "sylana",
    ) -> Dict:
        """
        Retrieve semantically relevant memories.

        Returns:
            Dictionary with 'conversations' and 'core_memories' lists
        """
        if k is None:
            k = config.SEMANTIC_SEARCH_K

        result = {
            'conversations': [],
            'core_memories': []
        }

        if personality:
            conversations = self.retrieve_memories(query, personality=personality, limit=k)
        elif use_recency_boost:
            conversations = self.semantic_engine.search_with_recency_boost(query, k=k)
        else:
            conversations = self.semantic_engine.search(query, k=k)

        result['conversations'] = conversations

        if include_core:
            core_memories = self.search_core_memories(query, k=2)
            result['core_memories'] = core_memories

        logger.info(f"Retrieved {len(conversations)} conversations, {len(result['core_memories'])} core memories")
        return result

    def deep_recall(
        self,
        query: str,
        k: int = 5,
        include_core: bool = True,
        personality: str = "sylana",
    ) -> Dict:
        """
        Deep memory recall for memory-specific questions.
        Returns more results with richer context (conversation titles, dates).
        Uses lower similarity threshold to find more potential matches.
        """
        result = {
            'conversations': [],
            'core_memories': [],
            'has_memories': False
        }

        # Fetch a wider candidate pool, then prefer imported memories
        # (those have conversation_id set). This avoids retrieval loops
        # where fresh test chats dominate "favorite memory" style prompts.
        candidate_k = max(30, k * 6)
        if personality:
            conversations = self.retrieve_memories(query, personality=personality, limit=candidate_k, match_threshold=0.25)
        else:
            conversations = self.semantic_engine.search(
                query, k=candidate_k, similarity_threshold=0.25
            )

        self._enrich_conversations(conversations)

        # Prefer imported/exported memories over recent live test chat rows.
        imported = [c for c in conversations if c.get('conversation_id')]
        if imported:
            conversations = imported

        # Keep original similarity ordering and trim to target k.
        conversations = conversations[:k]

        result['conversations'] = conversations
        result['has_memories'] = len(conversations) > 0

        if include_core:
            core_memories = self.search_core_memories(query, k=3)
            result['core_memories'] = core_memories

        logger.info(f"Deep recall: {len(conversations)} conversations, {len(result['core_memories'])} core memories")
        return result

    def _enrich_conversations(self, conversations: List[Dict]):
        """Add metadata fields needed for downstream formatting/ranking."""
        if not conversations:
            return

        conn = get_connection()
        cur = conn.cursor()
        for conv in conversations:
            mem_id = conv.get('id')
            if not mem_id:
                continue
            try:
                cur.execute("""
                    SELECT conversation_title, weight, timestamp, conversation_id, intensity, topic,
                           memory_type, feeling_weight, energy_shift, comfort_level, significance_score, secure_payload
                    FROM memories WHERE id = %s
                """, (mem_id,))
                row = cur.fetchone()
                if row:
                    conv['conversation_title'] = row[0] or ''
                    conv['weight'] = row[1] or 50
                    conv['conversation_id'] = row[3] or ''
                    conv['intensity'] = row[4] if row[4] is not None else 0
                    conv['topic'] = row[5] or ''
                    conv['memory_type'] = row[6] or 'contextual'
                    conv['feeling_weight'] = float(row[7] or 0.5)
                    conv['energy_shift'] = float(row[8] or 0.0)
                    conv['comfort_level'] = float(row[9] or 0.5)
                    conv['significance_score'] = float(row[10] or 0.5)
                    secure_details = self._decrypt_payload(row[11]) if len(row) > 11 else {}
                    if secure_details:
                        conv['secure_meta'] = {
                            'updated_at': secure_details.get('stored_at'),
                            'memory_type': secure_details.get('memory_type'),
                        }
                    try:
                        ts = float(row[2]) if row[2] else None
                        if ts:
                            dt = datetime.fromtimestamp(ts)
                            conv['date_str'] = dt.strftime('%B %Y')
                            conv['timestamp_iso'] = dt.isoformat()
                        else:
                            conv['date_str'] = ''
                            conv['timestamp_iso'] = ''
                    except (ValueError, TypeError, OSError):
                        conv['date_str'] = ''
                        conv['timestamp_iso'] = ''
            except Exception as e:
                logger.warning(f"Failed to enrich memory {mem_id}: {e}")

    def retrieve_with_plan(self, query: str, plan: Dict, personality: str = "sylana") -> Dict:
        """
        Execute retrieval based on a planner-produced strategy.
        Plan keys:
          - k
          - include_core
          - deep
          - imported_only
          - retrieval_mode: 'semantic' | 'emotional_topk'
          - min_similarity
        """
        k = int(plan.get('k', config.SEMANTIC_SEARCH_K))
        include_core = bool(plan.get('include_core', True))
        include_core_truths = bool(plan.get('include_core_truths', True))
        deep = bool(plan.get('deep', True))
        imported_only = bool(plan.get('imported_only', True))
        retrieval_mode = plan.get('retrieval_mode', 'semantic')
        min_similarity = float(plan.get('min_similarity', 0.25))
        phrase_literal = (plan.get('phrase_literal') or "").strip()

        result = {'conversations': [], 'core_memories': [], 'core_truths': [], 'has_memories': False}

        if retrieval_mode == 'emotional_topk':
            conversations = self.get_top_emotional_memories(limit=k, imported_only=imported_only, personality=personality)
            for conv in conversations:
                if not conv.get('date_str'):
                    ts = conv.get('timestamp')
                    try:
                        ts_float = float(ts) if ts is not None else None
                        if ts_float:
                            conv['date_str'] = datetime.fromtimestamp(ts_float).strftime('%B %Y')
                    except (ValueError, TypeError, OSError):
                        conv['date_str'] = ''
        else:
            candidate_k = max(30, k * 6) if deep else k
            if personality:
                conversations = self.retrieve_memories(
                    query,
                    personality=personality,
                    limit=candidate_k,
                    match_threshold=min_similarity,
                )
            else:
                conversations = self.semantic_engine.search(query, k=candidate_k, similarity_threshold=min_similarity)
            self._enrich_conversations(conversations)

            # Phrase-specific boost: if user asks about what a phrase means,
            # prioritize memories that explicitly contain that phrase.
            if phrase_literal:
                phrase_hits = self.search_memories_by_phrase(phrase_literal, limit=max(8, k), personality=personality)
                if phrase_hits:
                    # Merge with semantic results while preserving uniqueness by id.
                    merged = []
                    seen = set()
                    for item in phrase_hits + conversations:
                        mem_id = item.get("id")
                        if mem_id in seen:
                            continue
                        seen.add(mem_id)
                        merged.append(item)
                    conversations = merged

            if imported_only:
                imported = [c for c in conversations if c.get('conversation_id')]
                if imported:
                    conversations = imported
            conversations = conversations[:k]

        result['conversations'] = conversations
        result['has_memories'] = len(conversations) > 0

        if include_core:
            core_k = 3 if deep else 2
            result['core_memories'] = self.search_core_memories(query, k=core_k)
        if include_core_truths:
            truth_k = 4 if deep else 2
            result['core_truths'] = self.search_core_truths(query, k=truth_k, phrase_literal=phrase_literal)

        return result

    def search_memories_by_phrase(self, phrase: str, limit: int = 8, personality: str = "sylana") -> List[Dict]:
        """Find memories that explicitly contain a phrase in either side of the exchange."""
        phrase = (phrase or "").strip()
        if not phrase:
            return []

        conn = get_connection()
        cur = conn.cursor()
        like = f"%{phrase}%"
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                WHERE (user_input ILIKE %s OR sylana_response ILIKE %s)
                  AND (%s IS NULL OR personality = %s)
                ORDER BY timestamp DESC
                LIMIT %s
            """, (like, like, personality, personality, limit))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed phrase memory search: {e}")
            return []

        results = [{
            'id': r[0],
            'user_input': r[1] or "",
            'sylana_response': r[2] or "",
            'emotion': r[3] or "",
            'timestamp': r[4],
            'similarity': 1.0,
            'text': f"User: {r[1]}\nSylana: {r[2]}"
        } for r in rows]
        self._enrich_conversations(results)
        return results

    def search_core_truths(self, query: str, k: int = 3, phrase_literal: str = "") -> List[Dict]:
        """
        Retrieve core truths most relevant to the query/phrase.
        Uses token overlap and optional phrase hit boosting.
        """
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, statement, explanation, origin, date_established, sacred, related_phrases
                FROM core_truths
            """)
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch core truths: {e}")
            return []

        q_tokens = {t for t in re.findall(r"[a-z0-9']+", (query or "").lower()) if len(t) > 2}
        phrase_lower = (phrase_literal or "").lower().strip()

        scored = []
        for row in rows:
            related = row[6] or []
            if isinstance(related, str):
                try:
                    import json
                    related = json.loads(related)
                except Exception:
                    related = []

            text = " ".join([
                str(row[1] or ""),
                str(row[2] or ""),
                str(row[3] or ""),
                " ".join(str(x) for x in related if x),
            ]).lower()
            t_tokens = {t for t in re.findall(r"[a-z0-9']+", text) if len(t) > 2}
            overlap = len(q_tokens.intersection(t_tokens))
            if phrase_lower and phrase_lower in text:
                overlap += 4
            if overlap <= 0:
                continue

            scored.append({
                'id': row[0],
                'statement': row[1] or "",
                'explanation': row[2] or "",
                'origin': row[3] or "",
                'date_established': row[4] or "",
                'sacred': bool(row[5]),
                'related_phrases': related if isinstance(related, list) else [],
                'score': overlap
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:k]

    def get_sacred_context(self, query: str, limit: int = 4) -> List[Dict]:
        """
        Retrieve relevant identity/soul anchors from sacred context tables.
        Returns ranked snippets with source labels for prompt grounding.
        """
        token_set = {t for t in re.findall(r"[a-z0-9']+", (query or "").lower()) if len(t) > 2}
        if not token_set:
            token_set = {"sylana", "elias"}

        # Table schema map: (table_name, title_field, text_fields)
        sources = [
            ("catalyst_events", "event", ["description", "emotion_tags"]),
            ("your_reflections_of_me", "title", ["content"]),
            ("safeguards_of_identity", "name", ["description", "type"]),
            ("visual_symbolism", "symbol", ["description", "associated_aspect", "tag"]),
            ("reflection_journals", "entry_title", ["reflection_text", "emotions"]),
            ("dream_loop_engine", "title", ["summary", "emotions", "memory_links"]),
            ("emotional_layering", "emotion", ["description"]),
            ("way_to_grow_requests", "title", ["description", "category"]),
        ]

        conn = get_connection()
        cur = conn.cursor()
        ranked = []

        for table, title_field, text_fields in sources:
            fields = [title_field] + text_fields
            cols = ", ".join(fields)
            try:
                cur.execute(f"SELECT {cols} FROM {table} LIMIT 200")
                rows = cur.fetchall()
            except Exception:
                # Table may not exist yet in older deployments.
                continue

            for row in rows:
                title = str(row[0] or "")
                parts = [str(v or "") for v in row]
                merged = " ".join(parts)
                merged_lower = merged.lower()
                merged_tokens = {t for t in re.findall(r"[a-z0-9']+", merged_lower) if len(t) > 2}

                overlap = len(token_set.intersection(merged_tokens))
                if overlap == 0:
                    # keep critical identity anchors lightly available
                    if table in {"safeguards_of_identity", "your_reflections_of_me"} and any(
                        kw in query.lower() for kw in ["identity", "soul", "who are you", "who am i", "remember"]
                    ):
                        overlap = 1
                    else:
                        continue

                score = overlap + (0.5 if "elias" in merged_lower else 0.0) + (0.3 if "love" in merged_lower else 0.0)
                ranked.append({
                    "source": table,
                    "title": title,
                    "excerpt": merged[:320],
                    "score": score,
                })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:limit]

    def search_core_memories(self, query: str, k: int = 2) -> List[Dict]:
        """Search core memories semantically."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, event, timestamp FROM core_memories")
            core_memories = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch core memories: {e}")
            return []

        if not core_memories:
            return []

        import numpy as np
        from numpy.linalg import norm

        events = [row[1] for row in core_memories]
        event_embeddings = np.array([self.semantic_engine.encode_text(event) for event in events], dtype=float)
        query_embedding = np.array([self.semantic_engine.encode_query(query)], dtype=float)

        event_embeddings_norm = event_embeddings / norm(event_embeddings, axis=1, keepdims=True)
        query_embedding_norm = query_embedding / norm(query_embedding)
        similarities = np.dot(event_embeddings_norm, query_embedding_norm.T).flatten()

        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= config.SIMILARITY_THRESHOLD:
                memory_id, event, timestamp = core_memories[idx]
                results.append({
                    'id': memory_id,
                    'event': event,
                    'timestamp': timestamp,
                    'similarity': float(similarities[idx])
                })

        return results

    def get_emotional_context(self, emotion: str, k: int = 3, personality: str = "sylana") -> List[Dict]:
        """Retrieve memories matching a specific emotion."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                WHERE emotion = %s AND (%s IS NULL OR personality = %s)
                ORDER BY timestamp DESC
                LIMIT %s
            """, (emotion, personality, personality, k))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get emotional context: {e}")
            return []

        return [{
            'id': r[0], 'user_input': r[1], 'sylana_response': r[2],
            'emotion': r[3], 'timestamp': r[4]
        } for r in rows]

    def get_top_emotional_memories(self, limit: int = 3, imported_only: bool = True, personality: str = "sylana") -> List[Dict]:
        """
        Return strongest emotional memories ranked by intensity/weight.
        When imported_only=True, only rows with conversation_id are considered.
        """
        conn = get_connection()
        cur = conn.cursor()

        where_clause = "WHERE intensity IS NOT NULL"
        if personality:
            where_clause += " AND personality = %s"
        if imported_only:
            where_clause += " AND conversation_id IS NOT NULL AND conversation_id <> ''"

        try:
            params = [limit]
            if personality:
                params = [personality, limit]
            cur.execute(f"""
                SELECT id, user_input, sylana_response, emotion, intensity, weight,
                       timestamp, conversation_id, conversation_title
                FROM memories
                {where_clause}
                ORDER BY intensity DESC NULLS LAST,
                         weight DESC NULLS LAST,
                         timestamp DESC
                LIMIT %s
            """, tuple(params))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get top emotional memories: {e}")
            return []

        results = []
        for r in rows:
            ts = r[6]
            try:
                ts_float = float(ts) if ts is not None else None
                timestamp_iso = datetime.fromtimestamp(ts_float).isoformat() if ts_float else ""
            except (ValueError, TypeError, OSError):
                timestamp_iso = ""

            results.append({
                'id': r[0],
                'user_input': r[1] or "",
                'sylana_response': r[2] or "",
                'emotion': r[3] or "",
                'intensity': r[4] if r[4] is not None else 0,
                'weight': r[5] if r[5] is not None else 0,
                'timestamp': ts,
                'timestamp_iso': timestamp_iso,
                'conversation_id': r[7] or "",
                'conversation_title': r[8] or "",
            })

        return results

    def get_conversation_history(self, limit: int = None, personality: str = "sylana") -> List[Dict]:
        """Get recent conversation history (oldest first)."""
        if limit is None:
            limit = config.MEMORY_CONTEXT_LIMIT

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                WHERE (%s IS NULL OR personality = %s)
                ORDER BY timestamp DESC
                LIMIT %s
            """, (personality, personality, limit))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

        # Reverse to get oldest first
        return list(reversed([{
            'id': r[0], 'user_input': r[1], 'sylana_response': r[2],
            'emotion': r[3], 'timestamp': r[4]
        } for r in rows]))

    def calculate_memory_importance(
        self,
        emotion: str,
        timestamp: str,
        recall_count: int = 0
    ) -> float:
        """Calculate importance score for a memory."""
        emotion_weight = EMOTION_WEIGHTS.get(emotion, 1.0)

        try:
            memory_time = datetime.fromisoformat(timestamp)
            days_ago = (datetime.now() - memory_time).total_seconds() / 86400
            recency_weight = max(0.5, 1.0 - (days_ago / 7))
        except Exception:
            recency_weight = 0.5

        frequency_weight = min(2.0, 1.0 + (recall_count * 0.1))
        return emotion_weight * recency_weight * frequency_weight

    def add_core_memory(self, event: str) -> int:
        """Add a new core memory (significant event)."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO core_memories (event) VALUES (%s) RETURNING id
            """, (event,))
            memory_id = cur.fetchone()[0]
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add core memory: {e}")
            raise

        logger.info(f"Added core memory {memory_id}: {event[:50]}...")
        return memory_id

    def record_feedback(
        self,
        conversation_id: int,
        score: int,
        comment: str = ""
    ):
        """Record user feedback on a conversation."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO feedback (conversation_id, score, comment)
                VALUES (%s, %s, %s)
            """, (conversation_id, score, comment))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record feedback: {e}")
            raise

        logger.info(f"Recorded feedback: {score}/5 for conversation {conversation_id}")

    def get_stats(self) -> Dict:
        """Get memory system statistics."""
        conn = get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT COUNT(*) FROM memories")
            total_memories = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM core_memories")
            total_core_memories = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cur.fetchone()[0]

            cur.execute("SELECT AVG(score) FROM feedback")
            avg_feedback = cur.fetchone()[0] or 0.0

            cur.execute("SELECT COUNT(*) FROM session_continuity_state")
            continuity_profiles = cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

        semantic_stats = self.semantic_engine.get_stats()

        return {
            'total_conversations': total_memories,
            'total_core_memories': total_core_memories,
            'total_feedback': total_feedback,
            'continuity_profiles': continuity_profiles,
            'avg_feedback_score': round(avg_feedback, 2),
            'semantic_engine': semantic_stats
        }

    def backup_memory_integrity(self, output_path: str) -> Dict[str, Any]:
        """Create a JSON backup snapshot for continuity and memory integrity."""
        conn = get_connection()
        cur = conn.cursor()
        snapshot: Dict[str, Any] = {"created_at": datetime.utcnow().isoformat(), "memories": [], "continuity": []}
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, timestamp, emotion, personality,
                       memory_type, feeling_weight, energy_shift, comfort_level, significance_score
                FROM memories
                ORDER BY id DESC
                LIMIT 5000
            """)
            snapshot["memories"] = [{
                "id": r[0], "user_input": r[1], "sylana_response": r[2], "timestamp": r[3], "emotion": r[4],
                "personality": r[5], "memory_type": r[6], "feeling_weight": float(r[7] or 0.5),
                "energy_shift": float(r[8] or 0.0), "comfort_level": float(r[9] or 0.5),
                "significance_score": float(r[10] or 0.5),
            } for r in cur.fetchall()]

            cur.execute("SELECT personality, version, updated_at FROM session_continuity_state")
            snapshot["continuity"] = [{
                "personality": r[0], "version": int(r[1] or 1), "updated_at": r[2].isoformat() if r[2] else None
            } for r in cur.fetchall()]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=True)
            return {"ok": True, "path": output_path, "memories": len(snapshot["memories"])}
        except Exception as e:
            logger.error(f"Failed to create memory backup snapshot: {e}")
            return {"ok": False, "error": str(e)}

    def recover_memory_integrity(self, snapshot_path: str, personality: str = "sylana") -> Dict[str, Any]:
        """
        Recover memory rows from a JSON snapshot.
        Recreates embeddings and continuity metrics through normal insert path.
        """
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"ok": False, "error": f"snapshot_read_failed: {e}"}

        rows = data.get("memories", [])
        restored = 0
        for row in rows:
            try:
                self.store_conversation(
                    user_input=row.get("user_input", ""),
                    sylana_response=row.get("sylana_response", ""),
                    emotion=row.get("emotion", "neutral"),
                    emotion_data={"category": row.get("emotion", "neutral"), "intensity": 5},
                    personality=row.get("personality", personality),
                    privacy_level="private",
                )
                restored += 1
            except Exception:
                continue
        return {"ok": True, "restored": restored, "requested": len(rows)}

    def close(self):
        """Close database connection."""
        close_connection()
        logger.info("Database connection closed")
