"""
Sylana Vessel - Semantic Memory Search Engine
FAISS-based semantic search over conversation history
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import logging

from core.config_loader import config

logger = logging.getLogger(__name__)


class SemanticMemoryEngine:
    """
    Manages FAISS index for semantic similarity search over memories.
    Enables contextual retrieval of relevant past conversations.
    """

    def __init__(self, embedder: Optional[SentenceTransformer] = None):
        """
        Initialize semantic search engine

        Args:
            embedder: Pre-loaded SentenceTransformer model, or None to create new
        """
        if embedder is None:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        else:
            self.embedder = embedder

        self.index = None
        self.memory_texts = []
        self.memory_metadata = []  # Store (id, timestamp, emotion) for each memory
        self.dimension = None
        self.is_built = False

        logger.info("SemanticMemoryEngine initialized")

    def build_index(self, memories: List[Tuple[int, str, str, str, str]]):
        """
        Build FAISS index from conversation memories

        Args:
            memories: List of (id, user_input, sylana_response, emotion, timestamp)
        """
        if not memories:
            logger.warning("No memories provided to build index")
            self.index = None
            self.is_built = False
            return

        logger.info(f"Building FAISS index from {len(memories)} memories...")

        # Create conversation text representations
        self.memory_texts = []
        self.memory_metadata = []

        for mem_id, user_input, sylana_response, emotion, timestamp in memories:
            # Combine user input and response for semantic representation
            text = f"User: {user_input}\nSylana: {sylana_response}"
            self.memory_texts.append(text)
            self.memory_metadata.append({
                'id': mem_id,
                'user_input': user_input,
                'sylana_response': sylana_response,
                'emotion': emotion,
                'timestamp': timestamp
            })

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(
            self.memory_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Build FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))

        self.is_built = True
        logger.info(f"FAISS index built successfully with {len(self.memory_texts)} entries")

    def search(
        self,
        query: str,
        k: int = None,
        similarity_threshold: float = None
    ) -> List[dict]:
        """
        Search for semantically similar memories

        Args:
            query: Search query (user input)
            k: Number of results to return (default from config)
            similarity_threshold: Minimum similarity score (default from config)

        Returns:
            List of memory dictionaries with similarity scores
        """
        if not self.is_built or self.index is None:
            logger.warning("Index not built. Returning empty results.")
            return []

        if k is None:
            k = config.SEMANTIC_SEARCH_K
        if similarity_threshold is None:
            similarity_threshold = config.SIMILARITY_THRESHOLD

        # Generate query embedding
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Search FAISS index
        k_actual = min(k, len(self.memory_texts))
        distances, indices = self.index.search(query_embedding.astype('float32'), k_actual)

        # Convert L2 distances to similarity scores (0-1 range)
        # Lower distance = higher similarity
        # Use exponential decay: similarity = exp(-distance)
        similarities = np.exp(-distances[0])

        # Filter by threshold and format results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= similarity_threshold:
                memory = self.memory_metadata[idx].copy()
                memory['similarity'] = float(similarity)
                memory['text'] = self.memory_texts[idx]
                results.append(memory)

        logger.info(f"Found {len(results)} relevant memories (threshold: {similarity_threshold})")
        return results

    def search_with_recency_boost(
        self,
        query: str,
        k: int = None,
        recency_weight: float = 0.3
    ) -> List[dict]:
        """
        Search with recency bias - recent memories get boosted

        Args:
            query: Search query
            k: Number of results
            recency_weight: Weight for recency (0-1, default 0.3)

        Returns:
            List of memories sorted by combined similarity + recency score
        """
        results = self.search(query, k=k * 2, similarity_threshold=0.0)  # Get more candidates

        if not results:
            return []

        # Calculate recency scores (last 7 days get full weight)
        now = datetime.now()
        for memory in results:
            try:
                timestamp = datetime.fromisoformat(memory['timestamp'])
                hours_ago = (now - timestamp).total_seconds() / 3600
                days_ago = hours_ago / 24

                # Recency score: 1.0 for today, decays over 7 days
                recency_score = max(0.0, 1.0 - (days_ago / 7))
            except:
                recency_score = 0.0

            # Combined score
            memory['recency_score'] = recency_score
            memory['combined_score'] = (
                (1 - recency_weight) * memory['similarity'] +
                recency_weight * recency_score
            )

        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)

        # Return top k
        if k is None:
            k = config.SEMANTIC_SEARCH_K
        return results[:k]

    def search_by_emotion(
        self,
        query: str,
        emotion: str,
        k: int = None
    ) -> List[dict]:
        """
        Search for memories matching both query and emotion

        Args:
            query: Search query
            emotion: Target emotion (happy, sad, ecstatic, devastated, neutral)
            k: Number of results

        Returns:
            List of memories filtered by emotion
        """
        results = self.search(query, k=k * 3)  # Get more candidates

        # Filter by emotion
        emotion_matches = [r for r in results if r['emotion'] == emotion]

        if k is None:
            k = config.SEMANTIC_SEARCH_K

        return emotion_matches[:k]

    def get_similar_to_memory(self, memory_id: int, k: int = 5) -> List[dict]:
        """
        Find memories similar to a specific memory ID

        Args:
            memory_id: ID of the reference memory
            k: Number of similar memories to return

        Returns:
            List of similar memories (excluding the reference itself)
        """
        # Find the memory by ID
        ref_memory = None
        for idx, meta in enumerate(self.memory_metadata):
            if meta['id'] == memory_id:
                ref_memory = self.memory_texts[idx]
                break

        if ref_memory is None:
            logger.warning(f"Memory ID {memory_id} not found in index")
            return []

        # Search using the memory text as query
        results = self.search(ref_memory, k=k + 1)

        # Remove the reference memory itself
        results = [r for r in results if r['id'] != memory_id]

        return results[:k]

    def rebuild_if_stale(self, current_memory_count: int):
        """
        Rebuild index if memory count has changed significantly

        Args:
            current_memory_count: Current number of memories in database
        """
        if not self.is_built:
            return True  # Needs building

        indexed_count = len(self.memory_texts)
        difference = abs(current_memory_count - indexed_count)

        # Rebuild if difference > 10 memories or > 20%
        if difference > 10 or (indexed_count > 0 and difference / indexed_count > 0.2):
            logger.info(f"Index stale: {indexed_count} indexed vs {current_memory_count} in DB")
            return True

        return False

    def get_stats(self) -> dict:
        """Get statistics about the search engine"""
        return {
            'is_built': self.is_built,
            'dimension': self.dimension,
            'total_memories': len(self.memory_texts),
            'model': config.EMBEDDING_MODEL
        }


if __name__ == "__main__":
    # Test the semantic search engine
    logging.basicConfig(level=logging.INFO)

    # Example usage
    engine = SemanticMemoryEngine()

    # Mock data for testing
    test_memories = [
        (1, "How are you?", "I'm doing well, thank you!", "happy", "2025-12-20 10:00:00"),
        (2, "What's the weather?", "I don't have access to weather data", "neutral", "2025-12-20 11:00:00"),
        (3, "I'm feeling sad", "I'm here for you. What's troubling you?", "sad", "2025-12-21 14:00:00"),
        (4, "Tell me a joke", "Why did the AI go to therapy? Too many issues!", "happy", "2025-12-22 09:00:00"),
    ]

    engine.build_index(test_memories)

    # Test search
    results = engine.search("I'm not feeling great", k=2)
    print("\nSearch results for 'I'm not feeling great':")
    for r in results:
        print(f"  Similarity: {r['similarity']:.3f} - {r['user_input']}")

    # Test emotion-based search
    happy_results = engine.search_by_emotion("feeling", "happy", k=2)
    print("\nHappy memories about 'feeling':")
    for r in happy_results:
        print(f"  {r['user_input']}")

    print("\n" + str(engine.get_stats()))
