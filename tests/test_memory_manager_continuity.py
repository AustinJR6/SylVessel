from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parent.parent / "memory" / "memory_manager.py"
SPEC = importlib.util.spec_from_file_location("memory_manager_under_test", MODULE_PATH)
memory_manager = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader


class _StubConfig:
    MEMORY_ENCRYPTION_KEY = "test-key"
    APP_TIMEZONE = "America/Chicago"

    def __getattr__(self, name: str):
        return ""


class _StubSemanticMemoryEngine:
    def encode_query(self, query: str):
        return [0.0]


class _BootstrapCursor:
    def execute(self, sql, params=None):
        return None

    def fetchone(self):
        return (1,)


class _BootstrapConnection:
    def cursor(self):
        return _BootstrapCursor()


fake_core_pkg = types.ModuleType("core")
fake_memory_pkg = types.ModuleType("memory")
fake_config_loader = types.ModuleType("core.config_loader")
fake_config_loader.config = _StubConfig()
fake_semantic_search = types.ModuleType("memory.semantic_search")
fake_semantic_search.SemanticMemoryEngine = _StubSemanticMemoryEngine
fake_supabase_client = types.ModuleType("memory.supabase_client")
fake_supabase_client.get_connection = lambda: _BootstrapConnection()
fake_supabase_client.close_connection = lambda conn: None

sys.modules.setdefault("core", fake_core_pkg)
sys.modules["core.config_loader"] = fake_config_loader
sys.modules.setdefault("memory", fake_memory_pkg)
sys.modules["memory.semantic_search"] = fake_semantic_search
sys.modules["memory.supabase_client"] = fake_supabase_client

SPEC.loader.exec_module(memory_manager)
MemoryManager = memory_manager.MemoryManager


class _FakeConnection:
    def __init__(self):
        self.committed = False
        self.rolled_back = False

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class _FakeContinuityCursor:
    def __init__(self):
        self.last_sql = ""

    def execute(self, sql, params=None):
        self.last_sql = " ".join(str(sql).split())

    def fetchone(self):
        if "SELECT encrypted_state FROM session_continuity_state" in self.last_sql:
            return (b"encrypted",)
        return None

    def close(self):
        return None


class MemoryContinuityTests(unittest.TestCase):
    def setUp(self):
        self.manager = MemoryManager()

    def test_update_session_continuity_state_enriches_payload(self):
        conn = _FakeConnection()
        cur = _FakeContinuityCursor()
        existing_payload = {
            "user_state_markers": ["reflective", "hopeful"],
            "relationship_texture": ["building"],
            "care_signals": ["needs-clarity"],
            "communication_patterns": ["reflective"],
            "active_projects": ["deploy"],
            "preference_signals": ["concise"],
            "recent_relational_moments": [],
            "continuity_bridges": [],
            "recent_thread_ids": [2, 3],
        }

        with patch.object(self.manager, "_decrypt_payload", return_value=existing_payload):
            with patch.object(self.manager, "_encrypt_payload", return_value=b"encrypted-state"):
                with patch.object(
                    self.manager,
                    "list_upcoming_anniversaries",
                    return_value=[{"title": "Gus birthday", "days_until": 4, "importance": 8}],
                ):
                    with patch.object(
                        self.manager,
                        "_recent_milestone_candidates",
                        return_value=[{"title": "Shipped Vessel", "importance": 9, "quote": "we did it"}],
                    ):
                        payload = self.manager._update_session_continuity_state(
                            personality="sylana",
                            memory_id=11,
                            thread_id=4,
                            memory_type="relational",
                            emotion_data={"category": "sad"},
                            feeling_weight=0.82,
                            user_input="I'm exhausted and overwhelmed, and I still need to follow up on Gus birthday planning.",
                            sylana_response="Let's keep it gentle and make sure the birthday plan gets held.",
                            recent_layers={
                                "topics": ["birthday planning", "deploy"],
                                "entities": ["Gus", "Elias"],
                                "commitments": ["follow up on Gus birthday planning"],
                                "open_loops": [
                                    {
                                        "title": "Check in on Gus birthday plan",
                                        "priority": 0.9,
                                        "thread_id": 4,
                                        "due_hint": "tomorrow",
                                        "source_memory_id": 11,
                                    }
                                ],
                                "current_topic": "birthday planning",
                                "significance_score": 0.88,
                            },
                            conn=conn,
                            cur=cur,
                        )

        self.assertEqual(payload["last_emotion"], "sad")
        self.assertIn("tired", payload["user_state_markers"])
        self.assertIn("overloaded", payload["user_state_markers"])
        self.assertIn("building", payload["relationship_texture"])
        self.assertIn("family-centered", payload["care_signals"])
        self.assertIn("follow-through-needed", payload["care_signals"])
        self.assertLessEqual(len(payload["recent_relational_moments"]), 6)
        self.assertLessEqual(len(payload["continuity_bridges"]), 8)
        self.assertEqual(payload["recent_thread_ids"][0], 4)
        self.assertTrue(any(item.get("source_kind") == "open_loop" for item in payload["continuity_bridges"]))
        self.assertFalse(conn.rolled_back)

    def test_retrieve_tiered_context_uses_cross_thread_continuity_when_thread_is_sparse(self):
        with patch.object(self.manager, "_route_query_mode", return_value="continuity"):
            with patch.object(self.manager, "retrieve_memories", return_value=[]):
                with patch.object(self.manager, "_enrich_conversations"):
                    with patch.object(self.manager, "_score_episodes", return_value=[]):
                        with patch.object(self.manager, "_search_identity_core", return_value=[]):
                            with patch.object(self.manager, "_search_memory_facts", return_value=[]):
                                with patch.object(self.manager, "_search_fact_proposals", return_value=[]):
                                    with patch.object(self.manager, "_search_anniversaries", return_value=[]):
                                        with patch.object(self.manager, "_search_milestones", return_value=[]):
                                            with patch.object(
                                                self.manager,
                                                "_continuity_bundle",
                                                return_value={
                                                    "continuity_bridges": [
                                                        {
                                                            "summary": "Keep gentle follow-through on Gus birthday planning.",
                                                            "topic_key": "loop:gus-birthday",
                                                            "importance_score": 0.92,
                                                            "updated_at": "2026-03-28T12:00:00",
                                                            "thread_id": 12,
                                                        }
                                                    ]
                                                },
                                            ):
                                                with patch.object(self.manager, "_search_thread_working_memory", return_value={}):
                                                    with patch.object(self.manager, "_search_thread_summaries", return_value=[]):
                                                        with patch.object(self.manager, "_search_entities", return_value=[]):
                                                            with patch.object(self.manager, "_search_reflections_and_dreams", return_value={"reflections": [], "dreams": []}):
                                                                with patch.object(self.manager, "_record_query_audit"):
                                                                    with patch.object(
                                                                        self.manager,
                                                                        "list_open_loops",
                                                                        side_effect=[
                                                                            [],
                                                                            [
                                                                                {
                                                                                    "open_loop_id": 101,
                                                                                    "title": "Follow up on Gus birthday",
                                                                                    "thread_id": 12,
                                                                                    "due_hint": "tomorrow",
                                                                                }
                                                                            ],
                                                                        ],
                                                                    ):
                                                                        bundle = self.manager.retrieve_tiered_context(
                                                                            "What are we still carrying forward?",
                                                                            personality="sylana",
                                                                            thread_id=55,
                                                                        )

        self.assertEqual(bundle["query_mode"], "continuity")
        self.assertTrue(bundle["has_matches"])
        self.assertEqual(bundle["thread_summaries"][0]["window_kind"], "continuity_bridge")
        self.assertEqual(bundle["continuity_bridges"][0]["topic_key"], "loop:gus-birthday")
        self.assertEqual(bundle["open_loops"][0]["thread_scope"], "cross_thread")

    def test_build_nightly_carry_forward_briefs_summarizes_care_and_follow_through(self):
        briefs = self.manager._build_nightly_carry_forward_briefs(
            {
                "continuity": {
                    "user_state_markers": ["tired"],
                    "care_signals": ["follow-through-needed"],
                    "continuity_bridges": [{"summary": "Keep gentle follow-through on birthday planning."}],
                },
                "open_loops": [{"title": "Birthday planning"}],
                "anniversaries": [],
            }
        )

        self.assertIn("tired", briefs["care_brief"])
        self.assertIn("Birthday planning", briefs["follow_through_brief"])


if __name__ == "__main__":
    unittest.main()
