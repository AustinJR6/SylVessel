from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parent.parent / "memory" / "lysara_memory_manager.py"
SPEC = importlib.util.spec_from_file_location("lysara_memory_manager_under_test", MODULE_PATH)
lysara_memory_manager = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
fake_memory_pkg = types.ModuleType("memory")
fake_supabase_client = types.ModuleType("memory.supabase_client")


def _unused_pooled_cursor(commit=False):
    raise AssertionError("pooled_cursor should be patched per test")


fake_supabase_client.pooled_cursor = _unused_pooled_cursor
fake_memory_pkg.supabase_client = fake_supabase_client
sys.modules.setdefault("memory", fake_memory_pkg)
sys.modules["memory.supabase_client"] = fake_supabase_client
SPEC.loader.exec_module(lysara_memory_manager)
LysaraMemoryManager = lysara_memory_manager.LysaraMemoryManager


class _FakeCursor:
    def __init__(self, mapping):
        self.mapping = mapping
        self.description = []
        self._rows = []

    def execute(self, sql, params=None):
        sql_norm = " ".join(str(sql).split())
        for needle, payload in self.mapping.items():
            if needle in sql_norm:
                columns = payload["columns"]
                rows = payload["rows"]
                self.description = [(col, None, None, None, None, None, None) for col in columns]
                self._rows = [tuple(row.get(col) for col in columns) for row in rows]
                return
        raise AssertionError(f"Unexpected SQL: {sql_norm}")

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeCursorContext:
    def __init__(self, mapping):
        self.cursor = _FakeCursor(mapping)

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc, tb):
        return False


def _pooled_cursor_factory(mapping):
    def _factory(commit=False):
        return _FakeCursorContext(mapping)

    return _factory


class LysaraMemoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = LysaraMemoryManager()

    def test_parse_risk_markdown_preserves_types(self):
        payload = self.manager.parse_risk_markdown(
            """
            MAX_DAILY_LOSS_PCT: 3.0
            LIVE_AUTONOMOUS_TRADING_ENABLED: false
            ALLOWED_MARKETS: stocks, crypto
            APPROVAL_TTL_MINUTES: 30
            """
        )
        self.assertEqual(payload["max_daily_loss_pct"], 3.0)
        self.assertFalse(payload["live_autonomous_trading_enabled"])
        self.assertEqual(payload["allowed_markets"], ["stocks", "crypto"])
        self.assertEqual(payload["approval_ttl_minutes"], 30)

    def test_route_query_mode_detects_working_and_canonical(self):
        self.assertEqual(self.manager.route_query_mode("What is our current NVDA position right now?"), "working")
        self.assertEqual(self.manager.route_query_mode("What is the max total exposure rule?"), "canonical")
        self.assertEqual(self.manager.route_query_mode("Why did we exit AAPL yesterday?"), "episodic")

    def test_get_context_bundle_working_mode_only_runs_working_relevant_queries(self):
        now = datetime.now(timezone.utc)
        db_mapping = {
            "FROM lysara.strategy_profiles": {
                "columns": [
                    "strategy_key",
                    "name",
                    "description",
                    "market_scope",
                    "allowed_symbols_json",
                    "default_params_json",
                    "status",
                    "owner",
                    "source_ref",
                    "updated_at",
                ],
                "rows": [
                    {
                        "strategy_key": "MomentumStrategy",
                        "name": "Momentum",
                        "description": "trend",
                        "market_scope": ["stocks"],
                        "allowed_symbols_json": ["NVDA"],
                        "default_params_json": {},
                        "status": "active",
                        "owner": "lysara",
                        "source_ref": "ops",
                        "updated_at": now,
                    }
                ],
            },
            "FROM lysara.symbol_profiles": {
                "columns": [
                    "symbol",
                    "asset_class",
                    "market",
                    "liquidity_profile",
                    "volatility_profile",
                    "restriction_status",
                    "preferred_strategies_json",
                    "notes",
                    "updated_at",
                ],
                "rows": [
                    {
                        "symbol": "NVDA",
                        "asset_class": "equity",
                        "market": "stocks",
                        "liquidity_profile": "high",
                        "volatility_profile": "high",
                        "restriction_status": None,
                        "preferred_strategies_json": ["MomentumStrategy"],
                        "notes": "leader",
                        "updated_at": now,
                    }
                ],
            },
        }
        captured = {}

        def fake_working_state(**kwargs):
            captured.update(kwargs)
            return {"positions": [{"symbol": kwargs["symbol"]}]}

        with patch.object(lysara_memory_manager, "pooled_cursor", _pooled_cursor_factory(db_mapping)):
            with patch.object(self.manager, "get_working_state", side_effect=fake_working_state):
                with patch.object(self.manager, "list_open_loops", return_value={"items": [{"id": 1}]}):
                    with patch.object(self.manager, "list_review_queue", return_value={"items": [{"id": 2}]}):
                        with patch.object(self.manager, "_staleness_snapshot", return_value={"daemon": {"stale": False}}):
                            bundle = self.manager.get_context_bundle(query="What is our current NVDA situation right now?")

        self.assertEqual(bundle["query_mode"], "working")
        self.assertEqual(captured["symbol"], "NVDA")
        self.assertEqual(bundle["working_state"]["positions"][0]["symbol"], "NVDA")
        self.assertEqual(bundle["open_loops"]["loops"][0]["id"], 1)
        self.assertEqual(bundle["open_loops"]["review_queue"][0]["id"], 2)
        self.assertEqual(bundle["canonical_rules"]["strategies"][0]["strategy_key"], "MomentumStrategy")
        self.assertEqual(bundle["canonical_rules"]["risk_policies"], [])
        self.assertEqual(bundle["recent_operations"]["trade_decisions"], [])
        self.assertEqual(bundle["research_context"]["theses"], [])

    def test_get_context_bundle_canonical_mode_only_runs_canonical_queries(self):
        now = datetime.now(timezone.utc)
        db_mapping = {
            "FROM lysara.risk_policies": {
                "columns": [
                    "policy_key",
                    "name",
                    "scope",
                    "limits_json",
                    "approval_thresholds_json",
                    "autonomy_rules_json",
                    "status",
                    "effective_at",
                    "source_ref",
                    "updated_at",
                ],
                "rows": [
                    {
                        "policy_key": "default",
                        "name": "Default",
                        "scope": "global",
                        "limits_json": {"max_total_gross_exposure_pct": 100},
                        "approval_thresholds_json": {},
                        "autonomy_rules_json": {},
                        "status": "active",
                        "effective_at": now,
                        "source_ref": "RISK.md",
                        "updated_at": now,
                    }
                ],
            },
            "FROM lysara.portfolio_constraints": {
                "columns": ["portfolio_key", "constraint_type", "constraint_value_json", "status", "effective_at", "updated_at"],
                "rows": [{"portfolio_key": "main", "constraint_type": "gross", "constraint_value_json": {"max": 100}, "status": "active", "effective_at": now, "updated_at": now}],
            },
            "FROM lysara.operator_policies": {
                "columns": ["policy_key", "description", "tool_permissions_json", "approval_rules_json", "mutation_rules_json", "status", "updated_at"],
                "rows": [{"policy_key": "ops", "description": "operator", "tool_permissions_json": {}, "approval_rules_json": {}, "mutation_rules_json": {}, "status": "active", "updated_at": now}],
            },
            "FROM lysara.strategy_profiles": {
                "columns": ["strategy_key", "name", "description", "market_scope", "allowed_symbols_json", "default_params_json", "status", "owner", "source_ref", "updated_at"],
                "rows": [{"strategy_key": "MeanRevert", "name": "Mean Revert", "description": "fade", "market_scope": ["stocks"], "allowed_symbols_json": ["AAPL"], "default_params_json": {}, "status": "active", "owner": "lysara", "source_ref": "ops", "updated_at": now}],
            },
            "FROM lysara.symbol_profiles": {
                "columns": ["symbol", "asset_class", "market", "liquidity_profile", "volatility_profile", "restriction_status", "preferred_strategies_json", "notes", "updated_at"],
                "rows": [{"symbol": "AAPL", "asset_class": "equity", "market": "stocks", "liquidity_profile": "high", "volatility_profile": "medium", "restriction_status": None, "preferred_strategies_json": ["MeanRevert"], "notes": "", "updated_at": now}],
            },
        }

        with patch.object(lysara_memory_manager, "pooled_cursor", _pooled_cursor_factory(db_mapping)):
            with patch.object(self.manager, "get_working_state", side_effect=AssertionError("working state should not load")):
                with patch.object(self.manager, "list_open_loops", side_effect=AssertionError("open loops should not load")):
                    with patch.object(self.manager, "list_review_queue", side_effect=AssertionError("review queue should not load")):
                        with patch.object(self.manager, "_staleness_snapshot", return_value={"daemon": {"stale": False}}):
                            bundle = self.manager.get_context_bundle(query_mode="canonical", query="What are the standing rules?")

        self.assertEqual(bundle["query_mode"], "canonical")
        self.assertEqual(bundle["canonical_rules"]["risk_policies"][0]["policy_key"], "default")
        self.assertEqual(bundle["canonical_rules"]["strategies"][0]["strategy_key"], "MeanRevert")
        self.assertEqual(bundle["working_state"], {})
        self.assertEqual(bundle["open_loops"]["loops"], [])
        self.assertEqual(bundle["recent_operations"]["trade_decisions"], [])
        self.assertEqual(bundle["research_context"]["theses"], [])

    def test_get_context_bundle_episodic_mode_only_runs_recent_operation_queries(self):
        now = datetime.now(timezone.utc)
        db_mapping = {
            "FROM lysara.trade_decision_log": {
                "columns": [
                    "id",
                    "trade_ref",
                    "decision_type",
                    "symbol",
                    "strategy_key",
                    "market",
                    "regime_label",
                    "rationale",
                    "risk_snapshot_json",
                    "approval_state",
                    "review_item_id",
                    "decided_by",
                    "decided_at",
                    "final_status",
                    "execution_payload_json",
                    "source_ref",
                    "metadata_json",
                ],
                "rows": [
                    {
                        "id": 1,
                        "trade_ref": "trade-1",
                        "decision_type": "entry",
                        "symbol": "AAPL",
                        "strategy_key": "Breakout",
                        "market": "stocks",
                        "regime_label": "trend",
                        "rationale": "follow through",
                        "risk_snapshot_json": {},
                        "approval_state": "approved",
                        "review_item_id": None,
                        "decided_by": "lysara",
                        "decided_at": now,
                        "final_status": "filled",
                        "execution_payload_json": {},
                        "source_ref": "ops",
                        "metadata_json": {},
                    }
                ],
            },
            "FROM lysara.trade_performance": {
                "columns": ["metric_id", "trade_id", "source_trade_ref", "market", "symbol", "strategy_key", "strategy_name", "sector", "regime_label", "pnl", "pnl_pct", "win", "closed_at", "metadata_json"],
                "rows": [{"metric_id": 1, "trade_id": 1, "source_trade_ref": "trade-1", "market": "stocks", "symbol": "AAPL", "strategy_key": "Breakout", "strategy_name": "Breakout", "sector": "tech", "regime_label": "trend", "pnl": 120.0, "pnl_pct": 1.2, "win": True, "closed_at": now, "metadata_json": {}}],
            },
            "FROM lysara.operator_overrides": {
                "columns": ["id", "override_type", "symbol", "strategy_key", "market", "reason", "set_by", "set_at", "cleared_at", "source_ref", "payload_json"],
                "rows": [{"id": 1, "override_type": "pause", "symbol": "AAPL", "strategy_key": "Breakout", "market": "stocks", "reason": "vol spike", "set_by": "operator", "set_at": now, "cleared_at": None, "source_ref": "ops", "payload_json": {}}],
            },
            "FROM lysara.regime_history": {
                "columns": ["market", "symbol", "regime_label", "volatility_score", "trend_score", "confidence", "recommended_params_json", "applied", "source", "source_ref", "observed_at"],
                "rows": [{"market": "stocks", "symbol": "AAPL", "regime_label": "trend", "volatility_score": 0.7, "trend_score": 0.8, "confidence": 0.9, "recommended_params_json": {}, "applied": True, "source": "daemon", "source_ref": "ops", "observed_at": now}],
            },
        }

        with patch.object(lysara_memory_manager, "pooled_cursor", _pooled_cursor_factory(db_mapping)):
            with patch.object(self.manager, "get_working_state", side_effect=AssertionError("working state should not load")):
                with patch.object(self.manager, "list_open_loops", side_effect=AssertionError("open loops should not load")):
                    with patch.object(self.manager, "list_review_queue", side_effect=AssertionError("review queue should not load")):
                        with patch.object(self.manager, "_staleness_snapshot", return_value={"daemon": {"stale": False}}):
                            bundle = self.manager.get_context_bundle(query_mode="episodic", symbol="AAPL", market="stocks")

        self.assertEqual(bundle["query_mode"], "episodic")
        self.assertEqual(bundle["recent_operations"]["trade_decisions"][0]["trade_ref"], "trade-1")
        self.assertEqual(bundle["recent_operations"]["trade_performance"][0]["trade_id"], 1)
        self.assertEqual(bundle["canonical_rules"]["risk_policies"], [])
        self.assertEqual(bundle["research_context"]["notes"], [])

    def test_get_context_bundle_research_mode_only_runs_research_queries(self):
        now = datetime.now(timezone.utc)
        db_mapping = {
            "FROM lysara.research_notes": {
                "columns": ["note_type", "symbol", "strategy_key", "market", "title", "content", "tags_json", "recorded_at", "source_ref"],
                "rows": [{"note_type": "research", "symbol": "BTC-USD", "strategy_key": "MomentumStrategy", "market": "crypto", "title": "BTC note", "content": "Momentum intact", "tags_json": ["momentum"], "recorded_at": now, "source_ref": "ops"}],
            },
            "FROM lysara.market_theses": {
                "columns": ["thesis_key", "scope_type", "symbol", "sector", "strategy_key", "thesis_text", "confidence", "status", "supporting_refs_json", "source_note_id", "created_at", "updated_at"],
                "rows": [
                    {"thesis_key": f"t{i}", "scope_type": "symbol", "symbol": "BTC-USD", "sector": "crypto", "strategy_key": "MomentumStrategy", "thesis_text": str(i), "confidence": 0.5, "status": "active", "supporting_refs_json": [], "source_note_id": None, "created_at": now, "updated_at": now}
                    for i in range(1, 5)
                ],
            },
        }

        with patch.object(lysara_memory_manager, "pooled_cursor", _pooled_cursor_factory(db_mapping)):
            with patch.object(self.manager, "get_working_state", side_effect=AssertionError("working state should not load")):
                with patch.object(self.manager, "list_open_loops", side_effect=AssertionError("open loops should not load")):
                    with patch.object(self.manager, "list_review_queue", side_effect=AssertionError("review queue should not load")):
                        with patch.object(self.manager, "_staleness_snapshot", return_value={"daemon": {"stale": False}}):
                            bundle = self.manager.get_context_bundle(query_mode="research", symbol="BTC-USD", market="crypto")

        self.assertEqual(bundle["query_mode"], "research")
        self.assertEqual(bundle["research_context"]["notes"][0]["title"], "BTC note")
        self.assertEqual(len(bundle["research_context"]["theses"]), 4)
        self.assertEqual(bundle["working_state"], {})
        self.assertEqual(bundle["recent_operations"]["trade_decisions"], [])

    def test_get_context_bundle_explicit_sections_override_defaults(self):
        with patch.object(lysara_memory_manager, "pooled_cursor", side_effect=AssertionError("db should not be used")):
            with patch.object(self.manager, "get_working_state", return_value={"positions": [{"symbol": "TSLA"}]}):
                with patch.object(self.manager, "list_open_loops", side_effect=AssertionError("open loops should not load")):
                    with patch.object(self.manager, "list_review_queue", side_effect=AssertionError("review queue should not load")):
                        with patch.object(self.manager, "_staleness_snapshot", side_effect=AssertionError("staleness should not load")):
                            bundle = self.manager.get_context_bundle(
                                query_mode="research",
                                sections=["working_state"],
                                symbol="TSLA",
                            )

        self.assertEqual(bundle["query_mode"], "research")
        self.assertEqual(bundle["working_state"]["positions"][0]["symbol"], "TSLA")
        self.assertEqual(bundle["canonical_rules"]["strategies"], [])
        self.assertEqual(bundle["research_context"]["theses"], [])
        self.assertEqual(bundle["staleness"], {})


if __name__ == "__main__":
    unittest.main()
