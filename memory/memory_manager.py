"""
Sylana Vessel - Unified Memory Manager
Central interface for all memory operations with semantic search
"""

import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer

from core.config_loader import config
from memory.semantic_search import SemanticMemoryEngine

logger = logging.getLogger(__name__)


# Emotion weights for importance scoring
EMOTION_WEIGHTS = {
    "ecstatic": 2.0,      # High emotional intensity
    "devastated": 2.0,    # High emotional intensity
    "happy": 1.5,
    "sad": 1.5,
    "neutral": 1.0
}


class MemoryManager:
    """
    Unified memory management system combining:
    - SQLite storage
    - FAISS semantic search
    - Core memory retrieval
    - Importance scoring
    """

    def __init__(self, db_path: str = None):
        """
        Initialize memory manager

        Args:
            db_path: Path to SQLite database (default from config)
        """
        self.db_path = db_path or config.DB_PATH
        self.connection = None
        self.embedder = None
        self.semantic_engine = None

        # Initialize components
        self._connect_database()
        self._initialize_embedder()
        self._initialize_semantic_engine()

        logger.info(f"MemoryManager initialized with database: {self.db_path}")

    def _connect_database(self):
        """Establish database connection and ensure tables exist"""
        try:
            from pathlib import Path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self._ensure_tables_exist()
            logger.info("Database connection established")
        except Exception as e:
            logger.exception(f"Failed to connect to database: {e}")
            raise

    def _ensure_tables_exist(self):
        """Create required tables if they don't exist"""
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                sylana_response TEXT,
                emotion TEXT DEFAULT 'neutral',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS core_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                score INTEGER CHECK(score >= 1 AND score <= 5),
                comment TEXT DEFAULT '',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES memory(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_emotion ON memory(emotion)")
        self.connection.commit()
        logger.info("Database tables verified/created")

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
        # Build initial index
        self.rebuild_index()

    def rebuild_index(self):
        """Rebuild FAISS index from current database state"""
        logger.info("Rebuilding semantic search index...")
        memories = self._fetch_all_conversations()
        self.semantic_engine.build_index(memories)
        logger.info("Index rebuild complete")

    def _fetch_all_conversations(self) -> List[Tuple]:
        """Fetch all conversation memories from database"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, user_input, sylana_response, emotion, timestamp
            FROM memory
            ORDER BY timestamp ASC
        """)
        return cursor.fetchall()

    def store_conversation(
        self,
        user_input: str,
        sylana_response: str,
        emotion: str = "neutral"
    ) -> int:
        """
        Store a conversation turn with automatic importance scoring

        Args:
            user_input: User's message
            sylana_response: Sylana's response
            emotion: Detected emotion

        Returns:
            ID of inserted memory
        """
        cursor = self.connection.cursor()

        # Store conversation
        cursor.execute("""
            INSERT INTO memory (user_input, sylana_response, emotion)
            VALUES (?, ?, ?)
        """, (user_input, sylana_response, emotion))

        self.connection.commit()
        memory_id = cursor.lastrowid

        # Rebuild index if it's getting stale
        cursor.execute("SELECT COUNT(*) FROM memory")
        total_memories = cursor.fetchone()[0]

        if self.semantic_engine.rebuild_if_stale(total_memories):
            self.rebuild_index()

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
        Retrieve semantically relevant memories

        Args:
            query: Search query (current user input)
            k: Number of memories to retrieve
            include_core: Whether to include core memories
            use_recency_boost: Whether to boost recent memories

        Returns:
            Dictionary with 'conversations' and 'core_memories' lists
        """
        if k is None:
            k = config.SEMANTIC_SEARCH_K

        result = {
            'conversations': [],
            'core_memories': []
        }

        # Search conversation memories
        if use_recency_boost:
            conversations = self.semantic_engine.search_with_recency_boost(query, k=k)
        else:
            conversations = self.semantic_engine.search(query, k=k)

        result['conversations'] = conversations

        # Search core memories if requested
        if include_core:
            core_memories = self.search_core_memories(query, k=2)
            result['core_memories'] = core_memories

        logger.info(f"Retrieved {len(conversations)} conversations, {len(result['core_memories'])} core memories")
        return result

    def search_core_memories(self, query: str, k: int = 2) -> List[Dict]:
        """
        Search core memories semantically

        Args:
            query: Search query
            k: Number of core memories to return

        Returns:
            List of core memory dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT id, event, timestamp FROM core_memories")
        core_memories = cursor.fetchall()

        if not core_memories:
            return []

        # Encode core memory events
        events = [row[1] for row in core_memories]
        event_embeddings = self.embedder.encode(events, convert_to_numpy=True)

        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Calculate similarities (cosine similarity via dot product for normalized vectors)
        import numpy as np
        from numpy.linalg import norm

        # Normalize embeddings
        event_embeddings_norm = event_embeddings / norm(event_embeddings, axis=1, keepdims=True)
        query_embedding_norm = query_embedding / norm(query_embedding)

        # Calculate similarities
        similarities = np.dot(event_embeddings_norm, query_embedding_norm.T).flatten()

        # Get top k
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
        """
        Retrieve memories matching a specific emotion

        Args:
            emotion: Target emotion
            k: Number of memories to return

        Returns:
            List of memories with matching emotion
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, user_input, sylana_response, emotion, timestamp
            FROM memory
            WHERE emotion = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (emotion, k))

        return [dict(row) for row in cursor.fetchall()]

    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """
        Get recent conversation history

        Args:
            limit: Number of recent turns to retrieve

        Returns:
            List of conversation turns (oldest first)
        """
        if limit is None:
            limit = config.MEMORY_CONTEXT_LIMIT

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, user_input, sylana_response, emotion, timestamp
            FROM memory
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return list(reversed([dict(row) for row in rows]))  # Oldest first

    def calculate_memory_importance(
        self,
        emotion: str,
        timestamp: str,
        recall_count: int = 0
    ) -> float:
        """
        Calculate importance score for a memory

        Args:
            emotion: Detected emotion
            timestamp: Memory timestamp
            recall_count: Number of times recalled (future feature)

        Returns:
            Importance score (0.0-4.0+)
        """
        # Emotion weight
        emotion_weight = EMOTION_WEIGHTS.get(emotion, 1.0)

        # Recency weight (decay over 7 days)
        try:
            memory_time = datetime.fromisoformat(timestamp)
            hours_ago = (datetime.now() - memory_time).total_seconds() / 3600
            days_ago = hours_ago / 24
            recency_weight = max(0.5, 1.0 - (days_ago / 7))
        except:
            recency_weight = 0.5

        # Frequency weight (future feature - placeholder)
        frequency_weight = min(2.0, 1.0 + (recall_count * 0.1))

        # Combined importance
        importance = emotion_weight * recency_weight * frequency_weight

        return importance

    def add_core_memory(self, event: str) -> int:
        """
        Add a new core memory (significant event)

        Args:
            event: Description of the core memory

        Returns:
            ID of inserted core memory
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO core_memories (event)
            VALUES (?)
        """, (event,))
        self.connection.commit()

        memory_id = cursor.lastrowid
        logger.info(f"Added core memory {memory_id}: {event[:50]}...")
        return memory_id

    def record_feedback(
        self,
        conversation_id: int,
        score: int,
        comment: str = ""
    ):
        """
        Record user feedback on a conversation

        Args:
            conversation_id: ID of the conversation
            score: Rating (1-5)
            comment: Optional feedback comment
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO feedback (conversation_id, score, comment)
            VALUES (?, ?, ?)
        """, (conversation_id, score, comment))
        self.connection.commit()

        logger.info(f"Recorded feedback: {score}/5 for conversation {conversation_id}")

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        cursor = self.connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM memory")
        total_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM core_memories")
        total_core_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(score) FROM feedback")
        avg_feedback = cursor.fetchone()[0] or 0.0

        semantic_stats = self.semantic_engine.get_stats()

        return {
            'total_conversations': total_memories,
            'total_core_memories': total_core_memories,
            'total_feedback': total_feedback,
            'avg_feedback_score': round(avg_feedback, 2),
            'semantic_engine': semantic_stats
        }

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Test the memory manager
    logging.basicConfig(level=logging.INFO)

    manager = MemoryManager()

    # Get stats
    stats = manager.get_stats()
    print("\nMemory System Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test semantic search
    if stats['total_conversations'] > 0:
        print("\nTesting semantic search:")
        results = manager.recall_relevant("How are you feeling?", k=3)
        print(f"  Found {len(results['conversations'])} relevant conversations")
        for conv in results['conversations'][:2]:
            print(f"    - {conv['user_input'][:50]}... (similarity: {conv['similarity']:.3f})")

    manager.close()
