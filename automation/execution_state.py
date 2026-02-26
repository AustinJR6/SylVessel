from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExecutionState:
    workflow_id: str
    status: str = "pending"
    context: Dict[str, Any] = field(default_factory=dict)
