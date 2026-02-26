from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskNode:
    node_id: str
    name: str
    status: str = "pending"
    payload: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
