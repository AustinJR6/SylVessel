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

    def test_get_context_bundle_uses_symbol_from_query_and_truncates_theses_for_non_research(self):
        now = datetime.now(timezone.utc)
        db_mapping = {
            "FROM lysara.risk_policies": {
                "columns": ["policy_key", "name", "scope", "limits_json", "approval_thresholds_json", "autonomy_rules_json", "status", "effective_at", "source_ref", "updated_at"],
                "rows": [{"policy_key": "default", "name": "Default", "scope": "global", "limits_json": {"max_total_gross_exposure_pct": 100}, "approval_thresholds_json": {}, "autonomy_rules_json": {}, "status": "active", "effective_at": now, "source_ref": "RISK.md", "updated_at": now}],
            },
            "FROM lysara.strategy_profiles": {
                "columns": ["strategy_key", "name", "description", "market_scope", "allowed_symbols_json", "default_params_json", "status", "owner", "source_ref", "updated_at"],
                "rows": [{"strategy_key": "MomentumStrategy", "name": "Momentum", "description": "trend", "market_scope": ["stocks"], "allowed_symbols_json": ["NVDA"], "default_params_json": {}, "status": "active", "owner": "lysara", "source_ref": "ops", "updated_at": now}],
            },
            "FROM lysara.symbol_profiles": {
                "columns": ["symbol", "asset_class", "market", "liquidity_profile", "volatility_profile", "restriction_status", "preferred_strategies_json", "notes", "updated_at"],
                "rows": [{"symbol": "NVDA", "asset_class": "equity", "market": "stocks", "liquidity_profile": "high", "volatility_profile": "high", "restriction_status": None, "preferred_strategies_json": ["MomentumStrategy"], "notes": "leader", "updated_at": now}],
            },
            "FROM lysara.portfolio_constraints": {
                "columns": ["portfolio_key", "constraint_type", "constraint_value_json", "status", "effective_at", "updated_at"],
                "rows": [],
            },
            "FROM lysara.operator_policies": {
                "columns": ["policy_key", "description", "tool_permissions_json", "approval_rules_json", "mutation_rules_json", "status", "updated_at"],
                "rows": [],
            },
            "FROM lysara.trade_decision_log": {
                "columns": ["id", "trade_ref", "decision_type", "symbol", "strategy_key", "market", "regime_label", "rationale", "risk_snapshot_json", "approval_state", "review_item_id", "decided_by", "decided_at", "final_status", "execution_payload_json", "source_ref", "metadata_json"],
                "rows": [],
            },
            "FROM lysara.trade_performance": {
                "columns": ["metric_id", "trade_id", "source_trade_ref", "market", "symbol", "strategy_key", "strategy_name", "sector", "regime_label", "pnl", "pnl_pct", "win", "closed_at", "metadata_json"],
                "rows": [],
            },
            "FROM lysara.operator_overrides": {
                "columns": ["id", "override_type", "symbol", "strategy_key", "market", "reason", "set_by", "set_at", "cleared_at", "source_ref", "payload_json"],
                "rows": [],
            },
            "FROM lysara.regime_history": {
                "columns": ["market", "symbol", "regime_label", "volatility_score", "trend_score", "confidence", "recommended_params_json", "applied", "source", "source_ref", "observed_at"],
                "rows": [],
            },
            "FROM lysara.research_notes": {
                "columns": ["note_type", "symbol", "strategy_key", "market", "title", "content", "tags_json", "recorded_at", "source_ref"],
                "rows": [{"note_type": "research", "symbol": "NVDA", "strategy_key": "MomentumStrategy", "market": "stocks", "title": "AI demand", "content": "Strong demand", "tags_json": ["ai"], "recorded_at": now, "source_ref": "ops"}],
            },
            "FROM lysara.market_theses": {
                "columns": ["thesis_key", "scope_type", "symbol", "sector", "strategy_key", "thesis_text", "confidence", "status", "supporting_refs_json", "source_note_id", "created_at", "updated_at"],
                "rows": [
                    {"thesis_key": "t1", "scope_type": "symbol", "symbol": "NVDA", "sector": "tech", "strategy_key": "MomentumStrategy", "thesis_text": "one", "confidence": 0.7, "status": "active", "supporting_refs_json": [], "source_note_id": None, "created_at": now, "updated_at": now},
                    {"thesis_key": "t2", "scope_type": "symbol", "symbol": "NVDA", "sector": "tech", "strategy_key": "MomentumStrategy", "thesis_text": "two", "confidence": 0.8, "status": "active", "supporting_refs_json": [], "source_note_id": None, "created_at": now, "updated_at": now},
                    {"thesis_key": "t3", "scope_type": "symbol", "symbol": "NVDA", "sector": "tech", "strategy_key": "MomentumStrategy", "thesis_text": "three", "confidence": 0.9, "status": "active", "supporting_refs_json": [], "source_note_id": None, "created_at": now, "updated_at": now},
                    {"thesis_key": "t4", "scope_type": "symbol", "symbol": "NVDA", "sector": "tech", "strategy_key": "MomentumStrategy", "thesis_text": "four", "confidence": 0.6, "status": "active", "supporting_refs_json": [], "source_note_id": None, "created_at": now, "updated_at": now},
                ],
            },
        }
        captured = {}

        def fake_working_state(**kwargs):
            captured.update(kwargs)
            return {"positions": [{"symbol": kwargs["symbol"]}]}

        with patch.object(lysara_memory_manager, "pooled_cursor", _pooled_cursor_factory(db_mapping)):
            with patch.object(self.manager, "get_working_state", side_effect=fake_working_state):
                with patch.object(self.manager, "list_open_loops", return_value={"items": []}):
                    with patch.object(self.manager, "list_review_queue", return_value={"items": []}):
                        with patch.object(self.manager, "_staleness_snapshot", return_value={"status": {"stale": False}}):
                            bundle = self.manager.get_context_bundle(query="What is our current NVDA situation right now?")

        self.assertEqual(bundle["query_mode"], "working")
        self.assertEqual(captured["symbol"], "NVDA")
        self.assertEqual(bundle["working_state"]["positions"][0]["symbol"], "NVDA")
        self.assertEqual(len(bundle["research_context"]["theses"]), 3)

    def test_get_context_bundle_returns_full_theses_for_research_queries(self):
        now = datetime.now(timezone.utc)
        db_mapping = {
            "FROM lysara.risk_policies": {"columns": ["policy_key", "name", "scope", "limits_json", "approval_thresholds_json", "autonomy_rules_json", "status", "effective_at", "source_ref", "updated_at"], "rows": []},
            "FROM lysara.strategy_profiles": {"columns": ["strategy_key", "name", "description", "market_scope", "allowed_symbols_json", "default_params_json", "status", "owner", "source_ref", "updated_at"], "rows": []},
            "FROM lysara.symbol_profiles": {"columns": ["symbol", "asset_class", "market", "liquidity_profile", "volatility_profile", "restriction_status", "preferred_strategies_json", "notes", "updated_at"], "rows": []},
            "FROM lysara.portfolio_constraints": {"columns": ["portfolio_key", "constraint_type", "constraint_value_json", "status", "effective_at", "updated_at"], "rows": []},
            "FROM lysara.operator_policies": {"columns": ["policy_key", "description", "tool_permissions_json", "approval_rules_json", "mutation_rules_json", "status", "updated_at"], "rows": []},
            "FROM lysara.trade_decision_log": {"columns": ["id", "trade_ref", "decision_type", "symbol", "strategy_key", "market", "regime_label", "rationale", "risk_snapshot_json", "approval_state", "review_item_id", "decided_by", "decided_at", "final_status", "execution_payload_json", "source_ref", "metadata_json"], "rows": []},
            "FROM lysara.trade_performance": {"columns": ["metric_id", "trade_id", "source_trade_ref", "market", "symbol", "strategy_key", "strategy_name", "sector", "regime_label", "pnl", "pnl_pct", "win", "closed_at", "metadata_json"], "rows": []},
            "FROM lysara.operator_overrides": {"columns": ["id", "override_type", "symbol", "strategy_key", "market", "reason", "set_by", "set_at", "cleared_at", "source_ref", "payload_json"], "rows": []},
            "FROM lysara.regime_history": {"columns": ["market", "symbol", "regime_label", "volatility_score", "trend_score", "confidence", "recommended_params_json", "applied", "source", "source_ref", "observed_at"], "rows": []},
            "FROM lysara.research_notes": {"columns": ["note_type", "symbol", "strategy_key", "market", "title", "content", "tags_json", "recorded_at", "source_ref"], "rows": []},
            "FROM lysara.market_theses": {
                "columns": ["thesis_key", "scope_type", "symbol", "sector", "strategy_key", "thesis_text", "confidence", "status", "supporting_refs_json", "source_note_id", "created_at", "updated_at"],
                "rows": [
                    {"thesis_key": f"t{i}", "scope_type": "symbol", "symbol": "BTC-USD", "sector": "crypto", "strategy_key": "MomentumStrategy", "thesis_text": str(i), "confidence": 0.5, "status": "active", "supporting_refs_json": [], "source_note_id": None, "created_at": now, "updated_at": now}
                    for i in range(1, 5)
                ],
            },
        }
        with patch.object(lysara_memory_manager, "pooled_cursor", _pooled_cursor_factory(db_mapping)):
            with patch.object(self.manager, "get_working_state", return_value={}):
                with patch.object(self.manager, "list_open_loops", return_value={"items": []}):
                    with patch.object(self.manager, "list_review_queue", return_value={"items": []}):
                        with patch.object(self.manager, "_staleness_snapshot", return_value={}):
                            bundle = self.manager.get_context_bundle(query="What is the latest thesis on BTC-USD?")

        self.assertEqual(bundle["query_mode"], "research")
        self.assertEqual(len(bundle["research_context"]["theses"]), 4)


if __name__ == "__main__":
    unittest.main()
