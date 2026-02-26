from __future__ import annotations

import logging
from typing import Any, Dict

from .tool_contract import ToolRequest, ToolResult
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolRouter:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, request: ToolRequest, context: Dict[str, Any] | None = None) -> ToolResult:
        handler = self.registry.get(request.tool_name)
        if not handler:
            return ToolResult.error(f"Unknown tool: {request.tool_name}")
        try:
            return handler(request)
        except Exception as e:
            logger.exception("Tool execution failed: %s", e)
            return ToolResult.error(f"Tool execution failed: {e}")
