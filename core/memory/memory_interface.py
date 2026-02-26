from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .memory_types import MemoryRecord, RetrievalQuery, RetrievalResult


class MemoryInterface(ABC):
    @abstractmethod
    def store(self, memory: MemoryRecord) -> int:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        raise NotImplementedError

    @abstractmethod
    def history(self, identity: str, limit: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def consolidate(self, identity: Optional[str] = None, archive: bool = True) -> Dict[str, Any]:
        raise NotImplementedError
