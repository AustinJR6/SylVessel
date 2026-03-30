"""
Sylana Vessel - OpenRouter Model Adapter
OpenAI-compatible wrapper for OpenRouter chat completions with optional tool loops.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OpenRouterModel:
    """Thin wrapper over OpenRouter chat completions."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "Sylana Vessel",
        site_url: str = "",
        enable_web_search: bool = False,
        brave_api_key: str = "",
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment")
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.app_name = app_name or "Sylana Vessel"
        self.site_url = site_url or ""
        self.enable_web_search = bool(enable_web_search and brave_api_key)
        self.brave_api_key = brave_api_key or ""
        self.response_continuation_passes = max(0, int(os.getenv("RESPONSE_CONTINUATION_PASSES", "2") or "2"))
        self.external_tools_provider: Optional[Callable[[Optional[List[str]]], List[Dict[str, Any]]]] = None
        self.external_tool_runner: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None

    def set_external_tools(
        self,
        provider: Optional[Callable[[Optional[List[str]]], List[Dict[str, Any]]]] = None,
        runner: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.external_tools_provider = provider
        self.external_tool_runner = runner

    def _request_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    @staticmethod
    def _coerce_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean: List[Dict[str, Any]] = []
        for msg in messages or []:
            role = str(msg.get("role") or "user").strip()
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            content = msg.get("content")
            if role == "assistant" and msg.get("tool_calls"):
                clean.append({
                    "role": "assistant",
                    "content": content if isinstance(content, str) else "",
                    "tool_calls": msg.get("tool_calls"),
                })
                continue
            if role == "tool":
                clean.append({
                    "role": "tool",
                    "tool_call_id": str(msg.get("tool_call_id") or ""),
                    "content": content if isinstance(content, str) else json.dumps(content or {}),
                })
                continue
            if isinstance(content, str):
                text = content.strip()
                if text:
                    clean.append({"role": role, "content": text})
        return clean or [{"role": "user", "content": ""}]

    @staticmethod
    def _tools_to_openai_schema(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tool in tools:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description") or ""),
                        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
        return out

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {details[:400]}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc}") from exc

    def _tools(self, active_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        if self.external_tools_provider:
            try:
                tools.extend(self.external_tools_provider(active_tools) or [])
            except Exception as exc:
                logger.warning("OpenRouter external tools provider failed: %s", exc)
        return tools

    def _run_tool_call(self, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        if self.external_tool_runner:
            try:
                return self.external_tool_runner(name, tool_input or {})
            except Exception as exc:
                logger.warning("OpenRouter external tool '%s' failed: %s", name, exc)
                return {"error": f"external_tool_failed: {exc}", "tool": name}
        return {"error": f"Unknown tool: {name}"}

    @staticmethod
    def _extract_message(payload: Dict[str, Any]) -> Dict[str, Any]:
        choices = payload.get("choices") or []
        if not choices:
            return {}
        return (choices[0] or {}).get("message") or {}

    @staticmethod
    def _extract_finish_reason(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return str((choices[0] or {}).get("finish_reason") or "").strip().lower()

    @staticmethod
    def _join_text_segments(parts: List[str]) -> str:
        output = ""
        for raw in parts or []:
            piece = str(raw or "").strip()
            if not piece:
                continue
            if not output:
                output = piece
                continue
            if output[-1].isspace() or piece[0] in ",.;:!?)]}":
                output += piece
            else:
                output += " " + piece
        return output.strip()

    def _continue_truncated_response(
        self,
        *,
        convo: List[Dict[str, Any]],
        initial_text: str,
        response_payload: Dict[str, Any],
        max_tokens: int,
    ) -> str:
        if not initial_text or self.response_continuation_passes <= 0:
            return initial_text

        combined_parts = [initial_text]
        continuation_messages = list(convo or [])
        continuation_messages.append({"role": "assistant", "content": initial_text})
        current_payload = response_payload

        for _ in range(self.response_continuation_passes):
            if self._extract_finish_reason(current_payload) != "length":
                break
            continuation_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you left off. Do not restart, summarize, or repeat yourself. "
                        "Finish the same answer in plain text."
                    ),
                }
            )
            continuation_payload = self._post_chat(
                {
                    "model": self.model,
                    "messages": continuation_messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.9,
                }
            )
            next_message = self._extract_message(continuation_payload)
            next_content = next_message.get("content")
            next_text = next_content.strip() if isinstance(next_content, str) else ""
            if next_text:
                combined_parts.append(next_text)
                continuation_messages.append({"role": "assistant", "content": next_text})
            current_payload = continuation_payload

        return self._join_text_segments(combined_parts)

    def _finalize_empty_response(
        self,
        *,
        convo: List[Dict[str, Any]],
        message: Dict[str, Any],
        max_tokens: int,
    ) -> str:
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
        if tool_calls:
            convo.append(
                {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                function = call.get("function") or {}
                try:
                    parsed_args = json.loads(function.get("arguments") or "{}")
                except Exception:
                    parsed_args = {}
                tool_output = self._run_tool_call(str(function.get("name") or ""), parsed_args)
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "content": json.dumps(tool_output),
                    }
                )
        convo.append(
            {
                "role": "user",
                "content": (
                    "Answer Elias directly in plain text now using the tool results and context already gathered. "
                    "Do not call more tools."
                ),
            }
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": convo,
            "max_tokens": max_tokens,
            "temperature": 0.9,
        }
        response = self._post_chat(payload)
        final_message = self._extract_message(response)
        final_content = final_message.get("content")
        return final_content.strip() if isinstance(final_content, str) and final_content.strip() else ""

    def _generate_with_tools(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        active_tools: Optional[List[str]] = None,
    ) -> str:
        convo = [{"role": "system", "content": system_prompt.strip()}] if (system_prompt or "").strip() else []
        convo.extend(self._coerce_messages(messages))
        tools = self._tools(active_tools=active_tools)
        openai_tools = self._tools_to_openai_schema(tools)

        for _ in range(4):
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": convo,
                "max_tokens": max_tokens,
                "temperature": 0.9,
            }
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"

            response = self._post_chat(payload)
            message = self._extract_message(response)
            tool_calls = message.get("tool_calls") or []
            content = message.get("content")

            if not tool_calls:
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                    if self._extract_finish_reason(response) == "length":
                        logger.info("OpenRouterModel hit max_tokens; requesting continuation")
                        return self._continue_truncated_response(
                            convo=convo,
                            initial_text=text,
                            response_payload=response,
                            max_tokens=max_tokens,
                        )
                    return text
                return ""

            convo.append(
                {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                function = call.get("function") or {}
                try:
                    parsed_args = json.loads(function.get("arguments") or "{}")
                except Exception:
                    parsed_args = {}
                tool_output = self._run_tool_call(str(function.get("name") or ""), parsed_args)
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "content": json.dumps(tool_output),
                    }
                )

        logger.warning(
            "OpenRouterModel returned empty text after tool loop (tools=%s)",
            [tool.get("name") for tool in tools],
        )
        recovery_text = self._finalize_empty_response(
            convo=convo,
            message=message,
            max_tokens=max_tokens,
        )
        if recovery_text:
            return recovery_text
        logger.warning("OpenRouterModel recovery still produced empty text")
        return ""

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        active_tools: Optional[List[str]] = None,
    ) -> str:
        return self._generate_with_tools(system_prompt, messages, max_tokens, active_tools=active_tools)

    def generate_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        active_tools: Optional[List[str]] = None,
    ) -> Iterable[str]:
        full_text = self._generate_with_tools(system_prompt, messages, max_tokens, active_tools=active_tools)
        if not full_text:
            return
        for chunk in full_text.split(" "):
            if chunk:
                yield chunk + " "
