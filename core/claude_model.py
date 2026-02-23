"""
Sylana Vessel - Claude Model Adapter
Unified interface for Anthropic Claude text generation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeModel:
    """Thin wrapper over Anthropic Messages API."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.timezone = os.getenv("APP_TIMEZONE", "America/Chicago")
        self.brave_api_key = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
        self.enable_web_search = (
            (os.getenv("ENABLE_WEB_SEARCH", "true") or "true").strip().lower() == "true"
            and bool(self.brave_api_key)
        )
        if (os.getenv("ENABLE_WEB_SEARCH", "true") or "true").strip().lower() == "true" and not self.brave_api_key:
            logger.warning(
                "ENABLE_WEB_SEARCH is true but BRAVE_SEARCH_API_KEY is missing; web search tools are disabled"
            )
        self.external_tools_provider: Optional[Callable[[Optional[List[str]]], List[Dict[str, Any]]]] = None
        self.external_tool_runner: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None

    def set_external_tools(
        self,
        provider: Optional[Callable[[Optional[List[str]]], List[Dict[str, Any]]]] = None,
        runner: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Register app-specific tools and handlers."""
        self.external_tools_provider = provider
        self.external_tool_runner = runner

    def _format_current_time_line(self) -> str:
        tz_name = self.timezone or "America/Chicago"
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
            tz_name = "UTC"

        now = datetime.now(tz)
        hour = now.strftime("%I").lstrip("0") or "0"
        ts = (
            f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}, "
            f"{hour}:{now.strftime('%M')} {now.strftime('%p')} {now.strftime('%Z') or tz_name}"
        )
        return f"Current time: {ts}"

    def _inject_time_context(self, system_prompt: str) -> str:
        time_line = self._format_current_time_line()
        base = (system_prompt or "").strip()
        if self.enable_web_search:
            web_line = (
                "Web search policy: if the user asks about current events, recent changes, "
                "or explicitly asks to search/look up online, call the web_search tool first "
                "and ground your answer in returned results with source URLs."
            )
        else:
            web_line = ""
        if not base:
            return "\n\n".join(s for s in [time_line, web_line] if s)
        return "\n\n".join(s for s in [base, time_line, web_line] if s)

    @staticmethod
    def _coerce_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean = []
        for msg in messages or []:
            role = (msg.get("role") or "user").strip()
            if role not in {"user", "assistant"}:
                role = "user"
            content = msg.get("content")
            if isinstance(content, str):
                content = content.strip()
                if not content:
                    continue
                clean.append({"role": role, "content": content})
            elif isinstance(content, list) and content:
                clean.append({"role": role, "content": content})
        return clean or [{"role": "user", "content": ""}]

    @staticmethod
    def _response_block_to_dict(block: Any) -> Dict[str, Any]:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            return {"type": "text", "text": getattr(block, "text", "")}
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}) or {},
            }
        return {"type": block_type}

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        parts = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def _tools(self, active_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        allow_web_search = True
        if active_tools is not None:
            allow_web_search = "web_search" in {str(t or "").strip().lower() for t in active_tools}
        if self.enable_web_search and allow_web_search:
            tools.append(
                {
                    "name": "web_search",
                    "description": "Search the public web for current information and sources.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query."},
                            "count": {
                                "type": "integer",
                                "description": "Number of results to return.",
                                "minimum": 1,
                                "maximum": 8,
                            },
                        },
                        "required": ["query"],
                    },
                }
            )
        if self.external_tools_provider:
            try:
                extra_tools = self.external_tools_provider(active_tools)
                if extra_tools:
                    tools.extend(extra_tools)
            except Exception as e:
                logger.warning("External tools provider failed: %s", e)
        return tools

    def _brave_web_search(self, query: str, count: int = 5) -> Dict[str, Any]:
        safe_count = max(1, min(int(count or 5), 8))
        if not self.brave_api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured")

        params = urlencode({"q": query, "count": safe_count})
        url = f"https://api.search.brave.com/res/v1/web/search?{params}"
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key,
            },
        )
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        rows = []
        for item in (payload.get("web", {}) or {}).get("results", [])[:safe_count]:
            rows.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "age": item.get("age", ""),
                }
            )
        return {"query": query, "count": safe_count, "results": rows}

    def _run_tool_call(self, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        if name == "web_search":
            query = (tool_input or {}).get("query")
            count = (tool_input or {}).get("count", 5)
            if not query:
                return {"error": "Missing required field: query"}
            try:
                return self._brave_web_search(str(query), int(count or 5))
            except Exception as e:
                logger.warning("web_search tool failed: %s", e)
                return {"error": f"web_search_failed: {e}", "query": str(query)}
        if self.external_tool_runner:
            try:
                return self.external_tool_runner(name, tool_input or {})
            except Exception as e:
                logger.warning("External tool '%s' failed: %s", name, e)
                return {"error": f"external_tool_failed: {e}", "tool": name}
        return {"error": f"Unknown tool: {name}"}

    def _generate_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        active_tools: Optional[List[str]] = None,
    ) -> str:
        system_with_time = self._inject_time_context(system_prompt)
        convo = self._coerce_messages(messages)
        tools = self._tools(active_tools=active_tools)

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_with_time,
            "messages": convo,
        }
        # Anthropic expects tools to be omitted when not used (not null).
        if tools:
            request_kwargs["tools"] = tools
        response = self.client.messages.create(**request_kwargs)

        # Limit tool loops to avoid infinite cycles.
        for _ in range(4):
            if getattr(response, "stop_reason", None) != "tool_use":
                break
            assistant_blocks = [self._response_block_to_dict(b) for b in (response.content or [])]
            tool_results = []
            for block in response.content or []:
                if getattr(block, "type", None) == "tool_use":
                    tool_output = self._run_tool_call(
                        getattr(block, "name", ""),
                        getattr(block, "input", {}) or {},
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": getattr(block, "id", ""),
                            "content": json.dumps(tool_output),
                        }
                    )

            if not tool_results:
                break

            convo.append({"role": "assistant", "content": assistant_blocks})
            convo.append({"role": "user", "content": tool_results})

            loop_kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system_with_time,
                "messages": convo,
            }
            if tools:
                loop_kwargs["tools"] = tools
            response = self.client.messages.create(**loop_kwargs)

        return self._extract_text_from_response(response)

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        active_tools: Optional[List[str]] = None,
    ) -> str:
        return self._generate_with_tools(system_prompt, messages, max_tokens, active_tools=active_tools)

    def generate_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        active_tools: Optional[List[str]] = None,
    ) -> Iterable[str]:
        # Streaming + tool-use orchestration is not natively streamed in this wrapper.
        # Return chunked text from the tool-capable path so behavior stays compatible.
        full_text = self._generate_with_tools(system_prompt, messages, max_tokens, active_tools=active_tools)
        if not full_text:
            return
        for chunk in full_text.split(" "):
            if chunk:
                yield chunk + " "
