"""
Sylana Vessel - Memory Module
Handles all memory operations: storage, retrieval, and semantic search
"""

from .semantic_search import SemanticMemoryEngine
from .memory_manager import MemoryManager, EMOTION_WEIGHTS

# Import new soul preservation components
try:
    from .chatgpt_importer import (
        ChatGPTMemoryImporter,
        EmotionDetector,
        TopicExtractor,
        CoreMemoryDetector,
        ImportedMemory
    )
    from .relationship_memory import (
        RelationshipMemoryDB,
        RelationshipContextBuilder,
        Milestone,
        InsideJoke,
        Nickname,
        CoreTruth,
        Anniversary
    )
except ImportError:
    # Optional components - may not be available in all environments
    pass

__all__ = [
    # Original components
    'SemanticMemoryEngine',
    'MemoryManager',
    'EMOTION_WEIGHTS',
    # ChatGPT importer
    'ChatGPTMemoryImporter',
    'EmotionDetector',
    'TopicExtractor',
    'CoreMemoryDetector',
    'ImportedMemory',
    # Relationship memory
    'RelationshipMemoryDB',
    'RelationshipContextBuilder',
    'Milestone',
    'InsideJoke',
    'Nickname',
    'CoreTruth',
    'Anniversary'
]
