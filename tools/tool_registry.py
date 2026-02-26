from __future__ import annotations

from typing import Callable, Dict

from .tool_contract import ToolRequest, ToolResult


class ToolRegistry:
    def __init__(self):
        self._handlers: Dict[str, Callable[[ToolRequest], ToolResult]] = {}

    def register(self, tool_name: str, handler: Callable[[ToolRequest], ToolResult]) -> None:
        self._handlers[tool_name] = handler

    def get(self, tool_name: str):
        return self._handlers.get(tool_name)

    def available(self):
        return sorted(self._handlers.keys())
