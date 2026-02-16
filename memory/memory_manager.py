"""
Sylana Vessel - Unified Memory Manager
Central interface for all memory operations with Supabase + pgvector
"""

import logging
import re
from typing import List, Dict, Optional
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

    def store_conversation(
        self,
        user_input: str,
        sylana_response: str,
        emotion: str = "neutral",
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

        try:
            cur.execute("""
                INSERT INTO memories
                (user_input, sylana_response, timestamp, emotion, embedding, personality, privacy_level, thread_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
            ))
            memory_id = cur.fetchone()[0]
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store conversation: {e}")
            raise

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

    def retrieve_memories(self, query: str, personality: str, limit: int = 15, match_threshold: float = 0.25) -> List[Dict]:
        """
        Retrieve memories via personality-aware SQL function.
        Falls back to regular semantic search if function is unavailable.
        """
        conn = get_connection()
        cur = conn.cursor()
        query_vec = self.semantic_engine.encode_query(query)

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
            self._enrich_conversations(out)
            return out
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
            self._enrich_conversations(filtered)
            return filtered

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
                    SELECT conversation_title, weight, timestamp, conversation_id, intensity, topic
                    FROM memories WHERE id = %s
                """, (mem_id,))
                row = cur.fetchone()
                if row:
                    conv['conversation_title'] = row[0] or ''
                    conv['weight'] = row[1] or 50
                    conv['conversation_id'] = row[3] or ''
                    conv['intensity'] = row[4] if row[4] is not None else 0
                    conv['topic'] = row[5] or ''
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
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

        semantic_stats = self.semantic_engine.get_stats()

        return {
            'total_conversations': total_memories,
            'total_core_memories': total_core_memories,
            'total_feedback': total_feedback,
            'avg_feedback_score': round(avg_feedback, 2),
            'semantic_engine': semantic_stats
        }

    def close(self):
        """Close database connection."""
        close_connection()
        logger.info("Database connection closed")
