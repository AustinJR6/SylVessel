from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolRequest:
    tool_name: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    @classmethod
    def success(cls, data: Dict[str, Any], summary: str) -> "ToolResult":
        return cls(status="success", data=data, summary=summary)

    @classmethod
    def error(cls, message: str, data: Dict[str, Any] | None = None) -> "ToolResult":
        return cls(status="error", data=data or {}, summary=message)
