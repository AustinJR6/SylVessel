"""
Sylana Vessel - Unified Memory Manager
Central interface for all memory operations with Supabase + pgvector
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

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

    def __init__(self):
        self.embedder = None
        self.semantic_engine = None

        # Initialize components
        self._verify_connection()
        self._initialize_embedder()
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

    def _initialize_embedder(self):
        """Load sentence transformer for embeddings"""
        try:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.exception(f"Failed to load embedding model: {e}")
            raise

    def _initialize_semantic_engine(self):
        """Initialize semantic search engine"""
        self.semantic_engine = SemanticMemoryEngine(embedder=self.embedder)

    def rebuild_index(self):
        """No-op: pgvector handles indexing automatically."""
        logger.info("Index rebuild requested — pgvector handles this automatically")

    def store_conversation(
        self,
        user_input: str,
        sylana_response: str,
        emotion: str = "neutral"
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
                (user_input, sylana_response, timestamp, emotion, embedding)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                user_input,
                sylana_response,
                datetime.now().timestamp(),
                emotion,
                embedding
            ))
            memory_id = cur.fetchone()[0]
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store conversation: {e}")
            raise

        logger.info(f"Stored conversation {memory_id} with emotion: {emotion}")
        return memory_id

    def recall_relevant(
        self,
        query: str,
        k: int = None,
        include_core: bool = True,
        use_recency_boost: bool = True
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

        if use_recency_boost:
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
        include_core: bool = True
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
        conversations = self.semantic_engine.search(
            query, k=candidate_k, similarity_threshold=0.25
        )

        # Enrich with conversation titles and formatted timestamps
        if conversations:
            conn = get_connection()
            cur = conn.cursor()
            for conv in conversations:
                mem_id = conv.get('id')
                if mem_id:
                    try:
                        cur.execute("""
                            SELECT conversation_title, weight, timestamp, conversation_id
                            FROM memories WHERE id = %s
                        """, (mem_id,))
                        row = cur.fetchone()
                        if row:
                            conv['conversation_title'] = row[0] or ''
                            conv['weight'] = row[1] or 50
                            conv['conversation_id'] = row[3] or ''
                            try:
                                ts = float(row[2]) if row[2] else None
                                if ts:
                                    dt = datetime.fromtimestamp(ts)
                                    conv['date_str'] = dt.strftime('%B %Y')
                                else:
                                    conv['date_str'] = ''
                            except (ValueError, TypeError, OSError):
                                conv['date_str'] = ''
                    except Exception as e:
                        logger.warning(f"Failed to enrich memory {mem_id}: {e}")

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
        event_embeddings = self.embedder.encode(events, convert_to_numpy=True)
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

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

    def get_emotional_context(self, emotion: str, k: int = 3) -> List[Dict]:
        """Retrieve memories matching a specific emotion."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                WHERE emotion = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (emotion, k))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get emotional context: {e}")
            return []

        return [{
            'id': r[0], 'user_input': r[1], 'sylana_response': r[2],
            'emotion': r[3], 'timestamp': r[4]
        } for r in rows]

    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """Get recent conversation history (oldest first)."""
        if limit is None:
            limit = config.MEMORY_CONTEXT_LIMIT

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
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
