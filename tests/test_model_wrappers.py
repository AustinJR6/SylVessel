from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ClaudeModel = _load_module("claude_model_under_test", "core/claude_model.py").ClaudeModel
OpenRouterModel = _load_module("openrouter_model_under_test", "core/openrouter_model.py").OpenRouterModel


class _FakeClaudeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeClaudeResponse:
    def __init__(self, text: str, stop_reason: str):
        self.content = [_FakeClaudeBlock(text)]
        self.stop_reason = stop_reason


class ClaudeModelContinuationTests(unittest.TestCase):
    def test_generate_continues_when_claude_hits_max_tokens(self):
        calls = []
        responses = iter(
            [
                _FakeClaudeResponse("First part of the answer", "max_tokens"),
                _FakeClaudeResponse("and the rest of the answer.", "end_turn"),
            ]
        )

        def _create(**kwargs):
            calls.append(kwargs)
            return next(responses)

        model = ClaudeModel.__new__(ClaudeModel)
        model.client = SimpleNamespace(messages=SimpleNamespace(create=_create))
        model.model = "claude-test"
        model.timezone = "America/Chicago"
        model.enable_web_search = False
        model.response_continuation_passes = 2
        model.external_tools_provider = None
        model.external_tool_runner = None

        result = model.generate(
            system_prompt="system",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
            active_tools=[],
        )

        self.assertEqual(result, "First part of the answer and the rest of the answer.")
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            any(
                isinstance(message.get("content"), str)
                and "Continue exactly where you left off" in message.get("content", "")
                for message in calls[1]["messages"]
            )
        )
        self.assertNotIn("tools", calls[1])


class OpenRouterModelContinuationTests(unittest.TestCase):
    def test_generate_continues_when_openrouter_hits_length_limit(self):
        calls = []
        responses = iter(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "First part of the answer"},
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "and the rest of the answer."},
                        }
                    ]
                },
            ]
        )

        def _post(payload):
            calls.append(payload)
            return next(responses)

        model = OpenRouterModel.__new__(OpenRouterModel)
        model.model = "openrouter-test"
        model.response_continuation_passes = 2
        model.external_tools_provider = None
        model.external_tool_runner = None
        model._post_chat = _post

        result = model.generate(
            system_prompt="system",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
            active_tools=[],
        )

        self.assertEqual(result, "First part of the answer and the rest of the answer.")
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            any(
                isinstance(message.get("content"), str)
                and "Continue exactly where you left off" in message.get("content", "")
                for message in calls[1]["messages"]
            )
        )
        self.assertNotIn("tools", calls[1])


if __name__ == "__main__":
    unittest.main()
