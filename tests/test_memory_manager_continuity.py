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

    def cursor(self):
        raise NotImplementedError

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


class _StaticCursor:
    def __init__(self, fetchone_values=None, fetchall_values=None):
        self.fetchone_values = list(fetchone_values or [])
        self.fetchall_values = list(fetchall_values or [])
        self.executed_sql = []

    def execute(self, sql, params=None):
        self.executed_sql.append(" ".join(str(sql).split()))

    def fetchone(self):
        if self.fetchone_values:
            return self.fetchone_values.pop(0)
        return None

    def fetchall(self):
        if self.fetchall_values:
            return self.fetchall_values.pop(0)
        return []

    def close(self):
        return None


class _CursorConnection(_FakeConnection):
    def __init__(self, cursor):
        super().__init__()
        self._cursor = cursor

    def cursor(self):
        return self._cursor


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

    def test_get_conversation_history_falls_back_to_legacy_memories_ordering(self):
        cursor = _StaticCursor(
            fetchall_values=[
                [
                    (11, "We were talking about kite crypto", "Still tracking the token thesis.", "curious", 1774756141.0, 51, None, ""),
                ]
            ]
        )
        conn = _CursorConnection(cursor)

        with patch.object(self.manager, "_get_table_columns", return_value={"id", "user_input", "sylana_response", "emotion", "timestamp", "thread_id", "created_at", "personality"}):
            with patch.object(memory_manager, "get_connection", return_value=conn):
                history = self.manager.get_conversation_history(limit=1, personality="sylana", thread_id=51)

        self.assertEqual(history[0]["user_input"], "We were talking about kite crypto")
        self.assertIn("created_at", cursor.executed_sql[0])
        self.assertIn("to_timestamp(timestamp)", cursor.executed_sql[0])

    def test_store_conversation_omits_missing_temporal_columns_for_legacy_schema(self):
        cursor = _StaticCursor(fetchone_values=[(88,)])
        conn = _CursorConnection(cursor)
        self.manager.semantic_engine = types.SimpleNamespace(encode_text=lambda text: [0.0])

        temporal_context = {
            "recorded_at": "2026-03-29T03:49:01+00:00",
            "conversation_at": "2026-03-29T03:49:01+00:00",
            "user_local_date": "2026-03-28",
            "user_local_time": "22:49",
            "timezone_name": "America/Chicago",
            "turn_index": 7,
            "event_dates_json": [],
            "relative_time_labels": ["today"],
            "temporal_descriptor": "tonight",
        }

        with patch.object(memory_manager, "get_connection", return_value=conn):
            with patch.object(self.manager, "_get_table_columns", return_value={"user_input", "sylana_response", "timestamp", "emotion", "embedding", "personality", "privacy_level", "thread_id", "memory_type", "feeling_weight", "energy_shift", "comfort_level", "significance_score", "secure_payload"}):
                with patch.object(self.manager, "_classify_memory_type", return_value="contextual"):
                    with patch.object(self.manager, "_compute_feeling_weight", return_value=0.5):
                        with patch.object(self.manager, "_compute_energy_shift", return_value=0.0):
                            with patch.object(self.manager, "_compute_comfort_level", return_value=0.5):
                                with patch.object(self.manager, "_compute_significance_score", return_value=0.6):
                                    with patch.object(self.manager, "_infer_turn_index", return_value=7):
                                        with patch.object(self.manager, "_build_temporal_context", return_value=temporal_context):
                                            with patch.object(self.manager, "_encrypt_payload", return_value=b"payload"):
                                                with patch.object(self.manager, "_refresh_recent_memory_layers", return_value={}):
                                                    with patch.object(self.manager, "_update_session_continuity_state", return_value={}):
                                                        memory_id = self.manager.store_conversation(
                                                            user_input="kite crypto looks interesting",
                                                            sylana_response="Let's stay on the token thesis.",
                                                            emotion="curious",
                                                            emotion_data={"category": "curious"},
                                                            personality="sylana",
                                                            thread_id=51,
                                                        )

        self.assertEqual(memory_id, 88)
        insert_sql = next(sql for sql in cursor.executed_sql if "INSERT INTO memories" in sql)
        self.assertNotIn("recorded_at", insert_sql)
        self.assertNotIn("conversation_at", insert_sql)

    def test_upsert_memory_fact_preserves_explicit_user_correction(self):
        protected_fact = {
            "id": 9,
            "fact_key": "anniversary:gus_birthday",
            "fact_type": "birthday",
            "subject": "Gus",
            "value_json": {"date": "2022-01-13"},
            "normalized_text": "Gus birthday is January 13, 2022. Gus is Elias's son.",
            "confidence": 0.99,
            "importance": 1.85,
            "personality_scope": "shared",
            "source_kind": "user_correction",
            "source_ref": "user_correction",
        }
        with patch.object(self.manager, "_get_memory_fact", return_value=protected_fact):
            with patch.object(memory_manager, "get_connection", side_effect=AssertionError("db should not be touched")):
                fact = self.manager.upsert_memory_fact(
                    fact_key="anniversary:gus_birthday",
                    fact_type="birthday",
                    subject="Gus",
                    value_json={"date": "2021-01-13"},
                    normalized_text="Gus birthday is January 13, 2021. Gus is Elias's son.",
                    confidence=0.9,
                    personality_scope="shared",
                    source_kind="episode_backfill",
                )

        self.assertEqual(fact["value_json"]["date"], "2022-01-13")
        self.assertEqual(fact["source_kind"], "user_correction")


if __name__ == "__main__":
    unittest.main()
