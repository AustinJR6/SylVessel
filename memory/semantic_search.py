"""
Sylana Vessel - Semantic Memory Search Engine
pgvector-based semantic search over conversation history in Supabase
"""

from typing import List
from datetime import datetime
import logging

from openai import OpenAI

from core.config_loader import config

logger = logging.getLogger(__name__)


class SemanticMemoryEngine:
    """
    Semantic similarity search over memories using pgvector.
    Queries Supabase PostgreSQL directly — no local index needed.
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedding_model = config.EMBEDDING_MODEL
        self.embedding_dim = config.EMBEDDING_DIM
        logger.info(
            "SemanticMemoryEngine initialized (pgvector backend, OpenAI embeddings: %s, dim=%s)",
            self.embedding_model,
            self.embedding_dim,
        )

    def build_index(self, memories=None):
        """No-op: pgvector index is always live in Postgres."""
        pass

    def rebuild_if_stale(self, current_memory_count: int):
        """No-op: pgvector handles indexing automatically."""
        return False

    def encode_query(self, query: str) -> list:
        """Encode a query string into a vector."""
        text = (query or "").strip()
        if not text:
            text = "."
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[text],
            dimensions=self.embedding_dim,
        )
        return response.data[0].embedding

    def encode_text(self, text: str) -> list:
        """Encode any text into a vector for storage."""
        payload = (text or "").strip()
        if not payload:
            payload = "."
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[payload],
            dimensions=self.embedding_dim,
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        k: int = None,
        similarity_threshold: float = None
    ) -> List[dict]:
        """
        Search for semantically similar memories via pgvector cosine distance.

        Returns:
            List of memory dicts with 'similarity' scores (0-1, higher=better)
        """
        from memory.supabase_client import get_connection

        if k is None:
            k = config.SEMANTIC_SEARCH_K
        if similarity_threshold is None:
            similarity_threshold = config.SIMILARITY_THRESHOLD

        query_vec = self.encode_query(query)

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, query_vec, k))

            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
            conn.rollback()
            return []

        results = []
        for row in rows:
            mem_id, user_input, sylana_response, emotion, timestamp, similarity = row
            if similarity >= similarity_threshold:
                results.append({
                    'id': mem_id,
                    'user_input': user_input,
                    'sylana_response': sylana_response,
                    'emotion': emotion,
                    'timestamp': timestamp,
                    'similarity': float(similarity),
                    'text': f"User: {user_input}\nSylana: {sylana_response}"
                })

        logger.info(f"Found {len(results)} relevant memories (threshold: {similarity_threshold})")
        return results

    def search_with_recency_boost(
        self,
        query: str,
        k: int = None,
        recency_weight: float = 0.3
    ) -> List[dict]:
        """
        Search with recency bias — recent memories get boosted.
        Fetches 2*k candidates from pgvector, then applies recency formula.
        """
        if k is None:
            k = config.SEMANTIC_SEARCH_K

        # Get more candidates for re-ranking
        results = self.search(query, k=k * 2, similarity_threshold=0.0)

        if not results:
            return []

        now = datetime.now()
        for memory in results:
            try:
                ts = memory.get('timestamp')
                if ts:
                    dt = datetime.fromtimestamp(float(ts))
                    days_ago = (now - dt).total_seconds() / 86400
                    recency_score = max(0.0, 1.0 - (days_ago / 7))
                else:
                    recency_score = 0.0
            except (ValueError, TypeError, OSError):
                recency_score = 0.0

            memory['recency_score'] = recency_score
            memory['combined_score'] = (
                (1 - recency_weight) * memory['similarity'] +
                recency_weight * recency_score
            )

        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]

    def search_by_emotion(
        self,
        query: str,
        emotion: str,
        k: int = None
    ) -> List[dict]:
        """Search for memories matching both query and emotion."""
        from memory.supabase_client import get_connection

        if k is None:
            k = config.SEMANTIC_SEARCH_K

        query_vec = self.encode_query(query)

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL AND emotion = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, emotion, query_vec, k))

            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"pgvector emotion search failed: {e}")
            conn.rollback()
            return []

        results = []
        for row in rows:
            mem_id, user_input, sylana_response, emo, timestamp, similarity = row
            results.append({
                'id': mem_id,
                'user_input': user_input,
                'sylana_response': sylana_response,
                'emotion': emo,
                'timestamp': timestamp,
                'similarity': float(similarity),
                'text': f"User: {user_input}\nSylana: {sylana_response}"
            })

        return results

    def get_similar_to_memory(self, memory_id: int, k: int = 5) -> List[dict]:
        """Find memories similar to a specific memory ID."""
        from memory.supabase_client import get_connection

        conn = get_connection()
        cur = conn.cursor()
        try:
            # Get the embedding for the reference memory
            cur.execute("SELECT embedding FROM memories WHERE id = %s", (memory_id,))
            row = cur.fetchone()
            if not row or row[0] is None:
                logger.warning(f"Memory ID {memory_id} not found or has no embedding")
                return []

            ref_embedding = row[0]

            # Search for similar (excluding self)
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       1 - (embedding <=> %s) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL AND id != %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (ref_embedding, memory_id, ref_embedding, k))

            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"pgvector similar-to search failed: {e}")
            conn.rollback()
            return []

        results = []
        for row in rows:
            mem_id, user_input, sylana_response, emotion, timestamp, similarity = row
            results.append({
                'id': mem_id,
                'user_input': user_input,
                'sylana_response': sylana_response,
                'emotion': emotion,
                'timestamp': timestamp,
                'similarity': float(similarity),
                'text': f"User: {user_input}\nSylana: {sylana_response}"
            })

        return results

    def get_stats(self) -> dict:
        """Get statistics about the search engine."""
        from memory.supabase_client import get_connection

        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL")
            total = cur.fetchone()[0]
        except Exception:
            total = 0

        return {
            'is_built': True,
            'dimension': self.embedding_dim,
            'total_memories': total,
            'model': self.embedding_model,
            'backend': 'pgvector'
        }
