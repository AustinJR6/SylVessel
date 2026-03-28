from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SERVER_PATH = Path(__file__).resolve().parent.parent / "server.py"
INDEX_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _StubConfig:
    OPENAI_API_KEY = ""
    GITHUB_TOKEN = ""
    APP_TIMEZONE = "America/Chicago"

    def __getattr__(self, name: str):
        return ""


class _StubPromptEngineer:
    def build_system_message(self, personality_prompt: str, **kwargs):
        return personality_prompt


def _load_server_module():
    module_name = "server_under_test_chat_flow"
    if module_name in sys.modules:
        return sys.modules[module_name]

    backups = {}
    core_pkg = _stub_module("core")
    core_pkg.__path__ = []
    memory_pkg = _stub_module("memory")
    memory_pkg.__path__ = []
    stubs = {
        "core": core_pkg,
        "core.config_loader": _stub_module("core.config_loader", config=_StubConfig()),
        "core.prompt_engineer": _stub_module("core.prompt_engineer", PromptEngineer=_StubPromptEngineer),
        "core.claude_model": _stub_module("core.claude_model", ClaudeModel=type("ClaudeModel", (), {})),
        "core.openrouter_model": _stub_module("core.openrouter_model", OpenRouterModel=type("OpenRouterModel", (), {})),
        "core.lysara_ops": _stub_module(
            "core.lysara_ops",
            LysaraOpsClient=type("LysaraOpsClient", (), {}),
            LysaraOpsError=type("LysaraOpsError", (Exception,), {}),
        ),
        "memory": memory_pkg,
        "memory.supabase_client": _stub_module(
            "memory.supabase_client",
            get_connection=lambda: (_ for _ in ()).throw(AssertionError("unexpected database access")),
            init_connection_pool=lambda: None,
        ),
        "apscheduler": _stub_module("apscheduler"),
        "apscheduler.schedulers": _stub_module("apscheduler.schedulers"),
        "apscheduler.schedulers.asyncio": _stub_module(
            "apscheduler.schedulers.asyncio",
            AsyncIOScheduler=type("AsyncIOScheduler", (), {"__init__": lambda self, *args, **kwargs: None}),
        ),
        "apscheduler.triggers": _stub_module("apscheduler.triggers"),
        "apscheduler.triggers.cron": _stub_module(
            "apscheduler.triggers.cron",
            CronTrigger=type("CronTrigger", (), {"__init__": lambda self, *args, **kwargs: None}),
        ),
        "svix": _stub_module("svix"),
        "svix.webhooks": _stub_module(
            "svix.webhooks",
            Webhook=type("Webhook", (), {}),
            WebhookVerificationError=type("WebhookVerificationError", (Exception,), {}),
        ),
        "resend": _stub_module("resend"),
        "openai": _stub_module("openai", OpenAI=type("OpenAI", (), {"__init__": lambda self, *args, **kwargs: None})),
        "anthropic": _stub_module(
            "anthropic",
            APIConnectionError=type("APIConnectionError", (Exception,), {}),
            APIStatusError=type("APIStatusError", (Exception,), {}),
            APITimeoutError=type("APITimeoutError", (Exception,), {}),
            AuthenticationError=type("AuthenticationError", (Exception,), {}),
            BadRequestError=type("BadRequestError", (Exception,), {}),
            NotFoundError=type("NotFoundError", (Exception,), {}),
            RateLimitError=type("RateLimitError", (Exception,), {}),
        ),
    }
    try:
        for name, stub in stubs.items():
            backups[name] = sys.modules.get(name)
            sys.modules[name] = stub
        spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in backups.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


server = _load_server_module()


class _FakeJsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeDisconnectRequest:
    async def is_disconnected(self):
        return False


class ChatFlowTests(unittest.TestCase):
    def test_budget_prompt_sections_drops_low_priority_operational_summary(self):
        sections = [
            ("base_identity", "BASE " * 120, True),
            ("tool_policy", "TOOLS " * 100, True),
            ("working_memory", "WORKING " * 75, True),
            ("continuity", "CONTINUITY " * 55, True),
            ("factual_support", "SUPPORT " * 50, False),
            ("heartbeat", "HEARTBEAT " * 150, False),
        ]

        composed, dropped = server._budget_prompt_sections(sections, token_budget=720)

        self.assertIn("BASE", composed)
        self.assertIn("TOOLS", composed)
        self.assertIn("WORKING", composed)
        self.assertIn("CONTINUITY", composed)
        self.assertIn("SUPPORT", composed)
        self.assertIn("heartbeat", dropped)
        self.assertNotIn("HEARTBEAT", composed)

    def test_budget_recent_history_messages_keeps_most_recent_turns_and_truncates(self):
        recent_history = [
            {
                "user_input": f"user-{idx} " + ("u" * 600),
                "sylana_response": f"assistant-{idx} " + ("a" * 600),
            }
            for idx in range(1, 6)
        ]

        messages = server._budget_recent_history_messages(
            recent_history,
            token_budget=500,
            max_turns=4,
        )

        self.assertEqual(len(messages), 4)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant", "user", "assistant"])
        self.assertTrue(messages[0]["content"].startswith("user-4"))
        self.assertTrue(messages[2]["content"].startswith("user-5"))
        self.assertTrue(all(len(item["content"]) <= server.RECENT_HISTORY_MESSAGE_CHAR_LIMIT for item in messages))
        self.assertTrue(all(item["content"].endswith("...") for item in messages))

    def test_resolve_chat_request_context_creates_new_thread_when_missing(self):
        with patch.object(server, "create_chat_thread", return_value={"id": 77}) as create_thread:
            with patch.object(server, "_set_thread_tools") as set_thread_tools:
                ctx = server._resolve_chat_request_context(
                    raw_thread_id=None,
                    requested_tools=None,
                    personality="sylana",
                    user_input="hello there",
                )

        self.assertEqual(ctx["thread_id"], 77)
        self.assertEqual(ctx["active_tools"], server.DEFAULT_ACTIVE_TOOLS)
        self.assertTrue(ctx["created_new"])
        create_thread.assert_called_once()
        set_thread_tools.assert_not_called()

    def test_resolve_chat_request_context_reuses_valid_thread(self):
        with patch.object(server, "_thread_exists", return_value=True):
            with patch.object(server, "_get_thread_tools", return_value=["memories"]) as get_thread_tools:
                with patch.object(server, "_set_thread_tools") as set_thread_tools:
                    ctx = server._resolve_chat_request_context(
                        raw_thread_id="42",
                        requested_tools=None,
                        personality="sylana",
                        user_input="continue",
                    )

        self.assertEqual(ctx["thread_id"], 42)
        self.assertEqual(ctx["active_tools"], ["memories"])
        self.assertFalse(ctx["created_new"])
        get_thread_tools.assert_called_once_with(42)
        set_thread_tools.assert_called_once_with(42, ["memories"])

    def test_resolve_chat_request_context_rejects_invalid_explicit_thread(self):
        with patch.object(server, "_thread_exists", return_value=False):
            with self.assertRaises(server.ThreadContinuityError):
                server._resolve_chat_request_context(
                    raw_thread_id="999",
                    requested_tools=None,
                    personality="sylana",
                    user_input="resume",
                )

    def test_frontend_handles_session_event_and_continuity_error(self):
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("response.status === 409", html)
        self.assertIn("event.type === 'session'", html)
        self.assertIn("currentThreadId = event.data.thread_id;", html)
        self.assertIn("await refreshThreads(false);", html)


class ChatEndpointContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_endpoint_returns_409_for_invalid_explicit_thread(self):
        previous_ready = server.state.ready
        server.state.ready = True
        try:
            request = _FakeJsonRequest({"message": "hello", "thread_id": 999, "personality": "sylana"})
            with patch.object(server, "_thread_exists", return_value=False):
                response = await server.chat(request)
        finally:
            server.state.ready = previous_ready

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(response.status_code, 409)
        self.assertEqual(payload["error"], "thread_continuity_error")
        self.assertEqual(payload["requested_thread_id"], 999)

    async def test_chat_sync_endpoint_returns_409_for_invalid_explicit_thread(self):
        previous_ready = server.state.ready
        server.state.ready = True
        try:
            request = _FakeJsonRequest({"message": "hello", "thread_id": 999, "personality": "sylana"})
            with patch.object(server, "_thread_exists", return_value=False):
                response = await server.chat_sync(request)
        finally:
            server.state.ready = previous_ready

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(response.status_code, 409)
        self.assertEqual(payload["error"], "thread_continuity_error")
        self.assertEqual(payload["requested_thread_id"], 999)

    async def test_generate_response_stream_emits_session_event_first(self):
        stream = server.generate_response_stream(
            _FakeDisconnectRequest(),
            "hello",
            thread_id=123,
            personality="sylana",
            active_tools=["memories"],
            conversation_mode="default",
        )

        first_chunk = await stream.__anext__()
        await stream.aclose()

        self.assertTrue(first_chunk.startswith("data: "))
        payload = json.loads(first_chunk[len("data: ") :].strip())
        self.assertEqual(payload["type"], "session")
        self.assertEqual(payload["data"]["thread_id"], 123)
        self.assertEqual(payload["data"]["personality"], "sylana")
        self.assertEqual(payload["data"]["active_tools"], ["memories"])


if __name__ == "__main__":
    unittest.main()
