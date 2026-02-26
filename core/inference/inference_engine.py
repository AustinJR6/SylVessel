from __future__ import annotations

import logging
from typing import Dict, List, Optional

from core.claude_model import ClaudeModel
from core.config_loader import config
from core.ctransformers_model import CTransformersModelLoader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Unified inference facade for Claude API and quantized llama.cpp/ctransformers."""

    def __init__(self, mode: str = "claude"):
        self.mode = mode
        self.claude: Optional[ClaudeModel] = None
        self.quantized = None
        self._quantized_loader: Optional[CTransformersModelLoader] = None

        if mode == "quantized":
            model_path = getattr(config, "QUANTIZED_MODEL_PATH", "./models/llama-2-7b-chat.Q4_K_M.gguf")
            n_ctx = int(getattr(config, "QUANTIZED_N_CTX", 2048))
            self._quantized_loader = CTransformersModelLoader(model_path, context_length=n_ctx)
            self.quantized = self._quantized_loader.load_model()
        else:
            self.claude = ClaudeModel(api_key=config.ANTHROPIC_API_KEY, model=config.CLAUDE_MODEL)

    def set_tool_runtime(self, provider=None, runner=None) -> None:
        if self.claude:
            self.claude.set_external_tools(provider=provider, runner=runner)

    def generate(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        active_tools: Optional[List[str]] = None,
    ) -> str:
        if self.claude:
            return self.claude.generate(
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens or config.MAX_NEW_TOKENS,
                active_tools=active_tools,
            )

        if not self._quantized_loader:
            raise RuntimeError("Quantized inference backend unavailable")

        prompt = system_prompt + "\n\n" + "\n".join(
            f"{m.get('role','user')}: {m.get('content','')}" for m in messages
        )
        return self._quantized_loader.generate(
            prompt,
            max_tokens=max_tokens or config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )
