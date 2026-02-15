"""
Sylana Vessel - Claude Model Adapter
Unified interface for Anthropic Claude text generation.
"""

import os
from typing import Dict, Iterable, List

from anthropic import Anthropic


class ClaudeModel:
    """Thin wrapper over Anthropic Messages API."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    @staticmethod
    def _coerce_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        clean = []
        for msg in messages or []:
            role = (msg.get("role") or "user").strip()
            if role not in {"user", "assistant"}:
                role = "user"
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            clean.append({"role": role, "content": content})
        return clean or [{"role": "user", "content": ""}]

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], max_tokens: int = 4096) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=self._coerce_messages(messages),
        )
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def generate_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
    ) -> Iterable[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=self._coerce_messages(messages),
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
