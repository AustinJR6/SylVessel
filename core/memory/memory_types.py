from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    CORE_IDENTITY = "core_identity"
    RELATIONSHIP = "relationship"
    BELIEF = "belief"
    EMOTIONAL = "emotional"
    OPERATIONAL = "operational"
    DREAM = "dream"


@dataclass
class EmotionVector:
    valence: float
    arousal: float
    dominance: float
    category: str = "neutral"
    intensity: int = 5


@dataclass
class MemoryRecord:
    user_input: str
    sylana_response: str
    identity: str = "sylana"
    memory_type: MemoryType = MemoryType.OPERATIONAL
    emotion: EmotionVector = field(default_factory=lambda: EmotionVector(0.0, 0.5, 0.5))
    metadata: Dict[str, Any] = field(default_factory=dict)
    important: bool = False
    timestamp: Optional[float] = None


@dataclass
class RetrievalQuery:
    text: str
    identity: str = "sylana"
    memory_types: Optional[List[MemoryType]] = None
    limit: int = 8
    min_similarity: float = 0.25


@dataclass
class RetrievalResult:
    conversations: List[Dict[str, Any]]
    core_memories: List[Dict[str, Any]]
    scores: Dict[str, Any] = field(default_factory=dict)
