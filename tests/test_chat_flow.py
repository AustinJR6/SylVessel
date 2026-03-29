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


class _FakeHeaderRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


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

    def test_budget_recent_history_messages_skips_low_value_fallback_turns(self):
        recent_history = [
            {"user_input": "we were talking about kite", "sylana_response": "KITE is the token we were comparing."},
            {"user_input": "what changed?", "sylana_response": "I'm here with you. Say that again for me."},
        ]

        messages = server._budget_recent_history_messages(recent_history, token_budget=500, max_turns=4)

        self.assertEqual(messages[-1], {"role": "user", "content": "what changed?"})
        self.assertFalse(any("say that again" in item["content"].lower() for item in messages))

    def test_get_recent_thread_turn_history_prefers_persisted_chat_messages(self):
        chat_messages = [
            {"id": 1, "role": "user", "content": "we were talking about kite crypto", "turn": 7, "created_at": "2026-03-29T03:49:01"},
            {"id": 2, "role": "assistant", "content": "KITE looks like the token, not the coding tool", "turn": 7, "created_at": "2026-03-29T03:49:02"},
            {"id": 3, "role": "user", "content": "are competitors closing the gap", "turn": 8, "created_at": "2026-03-29T03:51:31"},
            {"id": 4, "role": "assistant", "content": "I'm here with you. Say that again for me.", "turn": 8, "created_at": "2026-03-29T03:51:32"},
        ]

        with patch.object(server, "get_chat_messages", return_value=chat_messages):
            history = server._get_recent_thread_turn_history(51, max_turns=2)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["user_input"], "we were talking about kite crypto")
        self.assertIn("token", history[0]["sylana_response"])
        self.assertEqual(history[1]["user_input"], "are competitors closing the gap")
        self.assertEqual(history[1]["sylana_response"], "")

    def test_load_recent_history_for_turn_prefers_thread_history_over_memory_history(self):
        fake_manager = types.SimpleNamespace(get_conversation_history=lambda **kwargs: (_ for _ in ()).throw(AssertionError("memory history should not be used")))
        previous_manager = getattr(server.state, "memory_manager", None)
        server.state.memory_manager = fake_manager
        try:
            with patch.object(server, "_get_recent_thread_turn_history", return_value=[{"user_input": "kite", "sylana_response": "token"}]):
                history = server._load_recent_history_for_turn(thread_id=51, personality="sylana", memories_active=True)
        finally:
            server.state.memory_manager = previous_manager

        self.assertEqual(history, [{"user_input": "kite", "sylana_response": "token"}])

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

    def test_frontend_quiet_inbox_ui_is_present(self):
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("Sylana Inbox", html)
        self.assertIn("/acknowledge", html)
        self.assertIn("/dismiss", html)
        self.assertIn("/approve", html)
        self.assertIn("/reject", html)
        self.assertIn("review item", html)
        self.assertIn("Repair Incidents", html)
        self.assertIn("/repairs/incidents", html)
        self.assertIn("investigateRepairIncident", html)

    def test_format_session_continuity_context_includes_bridges_and_care_signals(self):
        rendered = server._format_session_continuity_context(
            {
                "last_emotion": "tender",
                "emotional_baseline": "warm",
                "relationship_trust_level": 0.9,
                "conversation_momentum": "steady",
                "user_state_markers": ["tired"],
                "relationship_texture": ["building"],
                "care_signals": ["follow-through-needed"],
                "continuity_bridges": [
                    {"summary": "Keep gentle follow-through on the birthday plan."}
                ],
            }
        )

        self.assertIn("Recent user state markers", rendered)
        self.assertIn("Relationship texture", rendered)
        self.assertIn("Care signals", rendered)
        self.assertIn("Continuity bridges", rendered)
        self.assertIn("birthday plan", rendered)

    def test_run_prompt_session_isolated_does_not_store_memory_and_structures_note(self):
        with patch.object(server, "_create_work_session", return_value="session-1"):
            with patch.object(server, "_update_work_session"):
                with patch.object(server, "generate_response", return_value={"response": "Prepared outline"}) as generate_response:
                    with patch.object(server, "_enqueue_proactive_note", return_value={"note_id": "note-1"}) as enqueue_note:
                        with patch.object(server, "_fire_runtime_hooks"):
                            result = server._run_prompt_session(
                                entity="sylana",
                                prompt="Prepare something quiet.",
                                session_mode="isolated",
                                trigger_source="heartbeat",
                                note_kind="prep",
                                why_now="This thread is ripening.",
                                thread_id=42,
                                topic_key="prep:birthday",
                                memory_refs=["memory-1"],
                                importance_score=0.74,
                            )

        self.assertEqual(result["session_id"], "session-1")
        self.assertEqual(result["note"]["note_id"], "note-1")
        self.assertFalse(generate_response.call_args.kwargs["store_memory"])
        note_metadata = enqueue_note.call_args.kwargs["metadata"]
        self.assertEqual(note_metadata["note_kind"], "prep")
        self.assertEqual(note_metadata["thread_id"], 42)
        self.assertEqual(note_metadata["topic_key"], "prep:birthday")
        self.assertEqual(note_metadata["memory_refs"], ["memory-1"])

    def test_runtime_tool_specs_hide_memory_mutation_tools_for_plain_chat(self):
        fake_manager = types.SimpleNamespace(
            _is_explicit_correction=lambda text: False,
            _is_completion_signal=lambda text: False,
            list_open_loops=lambda **kwargs: [],
        )
        previous_manager = getattr(server.state, "memory_manager", None)
        server.state.memory_manager = fake_manager
        try:
            server._set_runtime_memory_tool_context(
                user_input="Tell me more about kite crypto.",
                personality="sylana",
                thread_id=51,
                conversation_mode="default",
                active_tools=["memories"],
            )
            specs = server._runtime_tool_specs(["memories"])
        finally:
            server._clear_runtime_memory_tool_context()
            server.state.memory_manager = previous_manager

        self.assertEqual([spec["name"] for spec in specs], [])

    def test_retry_empty_model_response_uses_plaintext_recovery(self):
        class _FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return "Recovered answer."

        model = _FakeModel()
        response = server._retry_empty_model_response(
            model=model,
            system_prompt="system",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
        )

        self.assertEqual(response, "Recovered answer.")
        self.assertEqual(model.calls[0]["active_tools"], [])
        self.assertIn("produced no visible reply", model.calls[0]["messages"][-1]["content"])


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

    async def test_list_proactive_notes_endpoint_supports_filters(self):
        with patch.object(server, "_list_proactive_notes", return_value=[{"note_id": "1"}]) as list_notes:
            response = await server.list_proactive_notes_endpoint(limit=12, status="pending", note_kind="care", thread_id=9, personality="sylana")

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["notes"], [{"note_id": "1"}])
        list_notes.assert_called_once_with(limit=12, status="pending", note_kind="care", thread_id=9, personality="sylana")

    async def test_get_proactive_queue_endpoint_returns_grouped_sections(self):
        queue_payload = {
            "summary": {"quiet_notes": 1, "approvals": 1, "prepared_work": 0, "total": 2},
            "sections": {"quiet_notes": [{"note_id": "1"}], "approvals": [{"note_id": "2"}], "prepared_work": []},
        }
        with patch.object(server, "_list_review_queue", return_value=queue_payload) as list_queue:
            response = await server.get_proactive_queue(limit=15, status="pending", personality="sylana", thread_id=7)

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["summary"]["total"], 2)
        self.assertEqual(payload["sections"]["approvals"][0]["note_id"], "2")
        list_queue.assert_called_once_with(limit=15, status="pending", personality="sylana", thread_id=7)

    async def test_acknowledge_and_dismiss_proactive_note_endpoints(self):
        with patch.object(server, "_set_proactive_note_status", return_value={"note_id": "1", "status": "surfaced"}) as set_status:
            response = await server.acknowledge_proactive_note("1")
        payload = json.loads(response.body.decode("utf-8"))
        self.assertTrue(payload["acknowledged"])
        self.assertEqual(payload["note"]["status"], "surfaced")
        set_status.assert_called_once_with("1", "surfaced")

        with patch.object(server, "_set_proactive_note_status", return_value={"note_id": "1", "status": "swallowed"}) as set_status:
            response = await server.dismiss_proactive_note("1")
        payload = json.loads(response.body.decode("utf-8"))
        self.assertTrue(payload["dismissed"])
        self.assertEqual(payload["note"]["status"], "swallowed")
        set_status.assert_called_once_with("1", "swallowed")

    async def test_update_proactive_note_endpoint_returns_updated_note(self):
        updated_note = {"note_id": "1", "title": "Corrected birthday note"}
        with patch.object(server, "_update_proactive_note", return_value=updated_note) as update_note:
            response = await server.update_proactive_note_endpoint(
                "1",
                server.ProactiveNoteUpdateRequest(
                    actor="mobile",
                    title="Corrected birthday note",
                    body="Gus is 1/13/2022 and Levi is 1/31/2024.",
                    why_now="User corrected the dates before review.",
                ),
            )

        payload = json.loads(response.body.decode("utf-8"))
        self.assertTrue(payload["updated"])
        self.assertEqual(payload["note"]["title"], "Corrected birthday note")
        update_note.assert_called_once()

    async def test_approve_endpoint_dispatches_review_action(self):
        note = {"note_id": "1", "action_kind": "outreach_research", "metadata": {}}
        with patch.object(server, "_get_proactive_note", side_effect=[note, note, note]):
            with patch.object(server, "_set_proactive_note_approval", return_value=note) as set_approval:
                with patch.object(server, "_dispatch_proactive_note_action", return_value={"ok": True}) as dispatch:
                    response = await server.approve_proactive_note("1")

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["execution"], {"ok": True})
        set_approval.assert_called_once()
        dispatch.assert_called_once()

    async def test_reject_endpoint_marks_note_rejected(self):
        note = {"note_id": "1", "approval_status": "rejected"}
        with patch.object(server, "_get_proactive_note", return_value=note):
            with patch.object(server, "_set_proactive_note_approval", return_value=note) as set_approval:
                response = await server.reject_proactive_note("1")

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["note"]["approval_status"], "rejected")
        set_approval.assert_called_once()

    async def test_autonomy_preferences_endpoints_round_trip(self):
        prefs = {
            "delivery_mode": "rare_push",
            "allowed_domains": {"internal": True, "outreach": True, "lysara": False},
            "quiet_hours": {"enabled": True, "start": "22:00", "end": "08:00", "timezone": "America/Chicago"},
            "daily_autonomous_cap": 4,
            "high_confidence_care_push_enabled": True,
        }
        with patch.object(server, "_get_autonomy_preferences", return_value=prefs):
            response = await server.get_autonomy_preferences_endpoint()
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["preferences"]["delivery_mode"], "rare_push")

        with patch.object(server, "_get_autonomy_preferences", return_value=prefs):
            with patch.object(server, "_set_autonomy_preferences", return_value=prefs) as set_prefs:
                response = await server.update_autonomy_preferences_endpoint(server.AutonomyPreferencesRequest(delivery_mode="rare_push"))
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["preferences"]["daily_autonomous_cap"], 4)
        set_prefs.assert_called_once()

    async def test_repair_incident_endpoints_round_trip(self):
        incident = {"incident_id": "inc-1", "status": "detected"}
        run = {"run_id": "run-1", "status": "proposed"}
        with patch.object(server, "_list_repair_incidents", return_value=[incident]) as list_incidents:
            response = await server.list_repair_incidents_endpoint(status="detected", repo="AustinJR6/SylVessel", limit=6)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["incidents"], [incident])
        list_incidents.assert_called_once_with(status="detected", repo="AustinJR6/SylVessel", limit=6)

        with patch.object(server, "_get_repair_incident", return_value=incident):
            with patch.object(server, "_list_repair_runs", return_value=[run]) as list_runs:
                response = await server.get_repair_incident_endpoint("inc-1")
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["incident"]["incident_id"], "inc-1")
        self.assertEqual(payload["runs"][0]["run_id"], "run-1")
        list_runs.assert_called_once_with(incident_id="inc-1", limit=20)

        with patch.object(server, "_get_repair_incident", return_value=incident):
            with patch.object(server, "_set_repair_incident_status", return_value={"incident_id": "inc-1", "status": "investigating"}) as set_status:
                with patch.object(server, "_record_repair_event") as record_event:
                    response = await server.investigate_repair_incident_endpoint(
                        "inc-1",
                        server.RepairInvestigateRequest(actor="operator", reason="please inspect"),
                    )
        payload = json.loads(response.body.decode("utf-8"))
        self.assertTrue(payload["queued"])
        self.assertEqual(payload["incident"]["status"], "investigating")
        set_status.assert_called_once()
        record_event.assert_called_once()

    async def test_repair_run_approve_endpoint_merges_and_updates_note(self):
        result = {"run": {"run_id": "run-1", "note_id": "note-1"}, "incident": {"incident_id": "inc-1"}}
        with patch.object(server, "_merge_repair_pull_request", return_value=result) as merge_pr:
            with patch.object(server, "_set_proactive_note_approval") as set_approval:
                with patch.object(server, "_update_proactive_note_execution") as update_exec:
                    response = await server.approve_repair_run_endpoint(
                        "run-1",
                        server.RepairRunDecisionRequest(actor="operator", reason="looks good"),
                    )
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["run"]["run_id"], "run-1")
        merge_pr.assert_called_once()
        set_approval.assert_called_once_with("note-1", True, actor="operator", reason="looks good")
        update_exec.assert_called_once_with("note-1", "merged")

    async def test_repair_run_reject_endpoint_blocks_note(self):
        result = {"run": {"run_id": "run-1", "note_id": "note-1"}, "incident": {"incident_id": "inc-1"}}
        with patch.object(server, "_reject_repair_run", return_value=result) as reject_run:
            with patch.object(server, "_set_proactive_note_approval") as set_approval:
                with patch.object(server, "_update_proactive_note_execution") as update_exec:
                    response = await server.reject_repair_run_endpoint(
                        "run-1",
                        server.RepairRunDecisionRequest(actor="operator", reason="too risky"),
                    )
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["run"]["run_id"], "run-1")
        reject_run.assert_called_once()
        set_approval.assert_called_once_with("note-1", False, actor="operator", reason="too risky")
        update_exec.assert_called_once_with("note-1", "blocked", stale_reason="too risky")

    async def test_repair_status_and_log_tail_endpoints(self):
        with patch.object(server, "_repair_status_summary", return_value={"open_incidents": 2, "pending_approvals": 1}):
            response = await server.repair_status_endpoint()
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["open_incidents"], 2)

        with patch.object(server, "_require_maintenance_read_access") as require_access:
            with patch.object(server, "_tail_log_lines", return_value=["line-1", "line-2"]) as tail:
                response = await server.repair_log_tail_endpoint(_FakeHeaderRequest({"X-Maintenance-Token": "token"}), lines=2)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["lines"], ["line-1", "line-2"])
        require_access.assert_called_once()
        tail.assert_called_once()


if __name__ == "__main__":
    unittest.main()
