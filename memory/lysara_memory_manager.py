from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from memory.supabase_client import pooled_cursor

logger = logging.getLogger(__name__)

QUERY_MODE_HINTS = {
    "working": {
        "current", "now", "today", "positions", "watchlist", "watchlists",
        "paused", "approval", "approvals", "what is happening", "status",
    },
    "canonical": {
        "rule", "rules", "policy", "policies", "limit", "limits",
        "allowed", "max", "maximum", "strategy definition", "constraint",
    },
    "episodic": {
        "why", "when did", "last trade", "yesterday", "what happened",
        "recent trade", "decision", "rationale", "exit",
    },
    "open_loop": {
        "watch", "revisit", "pending", "follow up", "waiting on",
        "review", "approval required",
    },
    "research": {
        "thesis", "outlook", "research", "catalyst", "catalysts",
        "journal", "narrative",
    },
}

CONTEXT_BUNDLE_SECTIONS = {
    "working_state",
    "open_loops",
    "canonical_rules",
    "recent_operations",
    "research_context",
    "staleness",
}

ALLOWED_QUERY_MODES = {"working", "canonical", "episodic", "open_loop", "research", "mixed"}

STALENESS_DEFAULTS = {
    "status": 180,
    "positions": 180,
    "portfolio": 180,
    "regime": 180,
    "exposure": 180,
    "sentiment": 180,
    "confluence": 180,
    "event_risk": 180,
    "research": 1800,
    "journal": 1800,
}

SYMBOL_TOKEN_RE = re.compile(r"\b[A-Za-z]{1,5}(?:-[A-Za-z]{2,6})?\b")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=True, sort_keys=True)


def _rows_to_dicts(cur) -> List[Dict[str, Any]]:
    cols = [col[0] for col in (cur.description or [])]
    items: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        payload: Dict[str, Any] = {}
        for idx, col in enumerate(cols):
            value = row[idx]
            payload[col] = value.isoformat() if isinstance(value, datetime) else value
        items.append(payload)
    return items


class LysaraMemoryManager:
    def __init__(self) -> None:
        self.staleness_defaults = dict(STALENESS_DEFAULTS)

    @staticmethod
    def parse_risk_markdown(markdown_text: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for raw_line in (markdown_text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key:
                continue
            low = value.lower()
            if low in {"true", "false"}:
                config[key] = low == "true"
                continue
            if "," in value:
                items = [item.strip() for item in value.split(",") if item.strip()]
                if items:
                    config[key] = items
                    continue
            try:
                config[key] = int(value)
                continue
            except Exception:
                pass
            try:
                config[key] = float(value)
                continue
            except Exception:
                pass
            config[key] = value
        return config

    @classmethod
    def route_query_mode(cls, query: str) -> str:
        text = (query or "").strip().lower()
        if not text:
            return "mixed"
        for mode in ("working", "canonical", "episodic", "open_loop", "research"):
            for hint in QUERY_MODE_HINTS[mode]:
                if hint in text:
                    return mode
        return "mixed"

    @staticmethod
    def normalize_query_mode(query_mode: Optional[str], fallback_query: str = "") -> str:
        clean = (query_mode or "").strip().lower()
        if clean in ALLOWED_QUERY_MODES:
            return clean
        return LysaraMemoryManager.route_query_mode(fallback_query)

    @staticmethod
    def normalize_context_sections(sections: Optional[Sequence[Any]]) -> List[str]:
        if not sections:
            return []
        raw_items: Sequence[Any]
        if isinstance(sections, str):
            raw_items = [item.strip() for item in sections.split(",")]
        else:
            raw_items = sections
        normalized: List[str] = []
        for raw in raw_items:
            clean = str(raw or "").strip().lower()
            if clean in CONTEXT_BUNDLE_SECTIONS and clean not in normalized:
                normalized.append(clean)
        return normalized

    @staticmethod
    def _empty_context_bundle(query_mode: str) -> Dict[str, Any]:
        return {
            "working_state": {},
            "open_loops": {"loops": [], "review_queue": []},
            "canonical_rules": {
                "risk_policies": [],
                "portfolio_constraints": [],
                "strategies": [],
                "symbol_profiles": [],
                "operator_policies": [],
            },
            "recent_operations": {
                "trade_decisions": [],
                "trade_performance": [],
                "operator_overrides": [],
                "regime_history": [],
            },
            "research_context": {
                "theses": [],
                "notes": [],
            },
            "query_mode": query_mode,
            "staleness": {},
        }

    @staticmethod
    def _default_sections_for_mode(query_mode: str) -> set[str]:
        if query_mode in {"working", "open_loop"}:
            return {"working_state", "open_loops", "canonical_rules", "staleness"}
        if query_mode == "canonical":
            return {"canonical_rules", "staleness"}
        if query_mode == "episodic":
            return {"recent_operations", "staleness"}
        if query_mode == "research":
            return {"research_context", "staleness"}
        return {"working_state", "open_loops", "canonical_rules", "recent_operations", "staleness"}

    @staticmethod
    def _extract_symbol_tokens(text: str) -> List[str]:
        raw_tokens = [token.upper() for token in SYMBOL_TOKEN_RE.findall(text or "")]
        filtered: List[str] = []
        stop = {
            "WHAT", "WHEN", "WITH", "FROM", "JUST", "NEED", "THIS",
            "THAT", "RISK", "RULE", "RULES", "WHY", "EXIT", "OPEN",
            "CLOSE", "PEND", "TODAY", "NOW", "STATUS", "IS", "OUR",
            "THE", "AND", "FOR", "ARE",
        }
        for token in raw_tokens:
            if token in stop or len(token) <= 1:
                continue
            filtered.append(token)
        return list(dict.fromkeys(filtered))

    def ensure_schema(self) -> Dict[str, Any]:
        statements = [
            "CREATE SCHEMA IF NOT EXISTS lysara",
            """
            CREATE TABLE IF NOT EXISTS lysara.sync_state (
                source_name TEXT PRIMARY KEY,
                last_success_at TIMESTAMPTZ,
                payload_updated_at TIMESTAMPTZ,
                stale_after_seconds INTEGER,
                error TEXT,
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.strategy_profiles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                strategy_key TEXT NOT NULL UNIQUE,
                name TEXT,
                description TEXT,
                market_scope JSONB NOT NULL DEFAULT '[]'::jsonb,
                allowed_symbols_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                default_params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'active',
                owner TEXT,
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.strategy_revisions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                strategy_profile_id UUID REFERENCES lysara.strategy_profiles(id) ON DELETE CASCADE,
                revision_no INTEGER NOT NULL,
                params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                change_summary TEXT,
                change_reason TEXT,
                changed_by TEXT,
                approved_by TEXT,
                approved_at TIMESTAMPTZ,
                source_ref TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(strategy_profile_id, revision_no)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.risk_policies (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                policy_key TEXT NOT NULL UNIQUE,
                name TEXT,
                scope TEXT NOT NULL DEFAULT 'global',
                limits_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                approval_thresholds_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                autonomy_rules_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'draft',
                effective_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.risk_policy_revisions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                risk_policy_id UUID REFERENCES lysara.risk_policies(id) ON DELETE CASCADE,
                revision_no INTEGER NOT NULL,
                limits_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                approval_thresholds_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                autonomy_rules_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                change_summary TEXT,
                change_reason TEXT,
                changed_by TEXT,
                approved_by TEXT,
                approved_at TIMESTAMPTZ,
                source_ref TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(risk_policy_id, revision_no)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.portfolio_constraints (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                portfolio_key TEXT NOT NULL DEFAULT 'default',
                constraint_type TEXT NOT NULL,
                constraint_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'active',
                effective_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.symbol_profiles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                symbol TEXT NOT NULL,
                asset_class TEXT,
                market TEXT NOT NULL DEFAULT 'all',
                liquidity_profile TEXT,
                volatility_profile TEXT,
                restriction_status TEXT,
                preferred_strategies_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                notes TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(symbol, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.operator_policies (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                policy_key TEXT NOT NULL UNIQUE,
                description TEXT,
                tool_permissions_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                approval_rules_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                mutation_rules_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'active',
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.portfolio_working_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                portfolio_key TEXT NOT NULL DEFAULT 'default',
                market TEXT NOT NULL DEFAULT 'all',
                as_of TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                paused BOOLEAN,
                pause_reason TEXT,
                simulation_mode BOOLEAN,
                live_trading_enabled BOOLEAN,
                autonomous_mode BOOLEAN,
                total_equity DOUBLE PRECISION,
                cash DOUBLE PRECISION,
                buying_power DOUBLE PRECISION,
                portfolio_value DOUBLE PRECISION,
                gross_exposure_pct DOUBLE PRECISION,
                heat_score DOUBLE PRECISION,
                runtime_flags_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                source_ref TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(portfolio_key, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.active_positions_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                account_scope TEXT NOT NULL DEFAULT 'default',
                market TEXT NOT NULL DEFAULT 'all',
                symbol TEXT NOT NULL,
                side TEXT,
                quantity DOUBLE PRECISION,
                entry_price DOUBLE PRECISION,
                mark_price DOUBLE PRECISION,
                unrealized_pnl DOUBLE PRECISION,
                strategy_key TEXT,
                stop_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                target_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                risk_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'open',
                as_of TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                source_ref TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(account_scope, market, symbol)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.market_working_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_name TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'all',
                symbol TEXT NOT NULL DEFAULT '*',
                as_of TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                top_signals_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                catalysts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                risk_posture TEXT,
                feed_freshness JSONB NOT NULL DEFAULT '{}'::jsonb,
                confidence DOUBLE PRECISION,
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(source_name, market, symbol)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.current_regime_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                market TEXT NOT NULL UNIQUE,
                regime_label TEXT,
                volatility_score DOUBLE PRECISION,
                trend_score DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                recommended_params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.active_watchlists (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                watchlist_key TEXT NOT NULL DEFAULT 'default',
                symbol TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'all',
                priority DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                thesis_ref TEXT,
                trigger_conditions_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'active',
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(watchlist_key, symbol, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.strategy_runtime_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                strategy_key TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'all',
                status TEXT NOT NULL DEFAULT 'unknown',
                paused BOOLEAN NOT NULL DEFAULT FALSE,
                pause_reason TEXT,
                runtime_params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                autonomy_mode TEXT,
                symbol_controls_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(strategy_key, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.review_queue (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                queue_type TEXT NOT NULL,
                symbol TEXT,
                strategy_key TEXT,
                market TEXT,
                priority DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                title TEXT NOT NULL,
                details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending',
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                resolved_at TIMESTAMPTZ,
                resolution_note TEXT,
                source_ref TEXT,
                external_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                UNIQUE(queue_type, source_ref)
            )
            """,
        ]
        phase2_plus = [
            """
            CREATE TABLE IF NOT EXISTS lysara.trade_decision_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                trade_ref TEXT,
                decision_type TEXT NOT NULL,
                symbol TEXT,
                strategy_key TEXT,
                market TEXT,
                regime_label TEXT,
                signal_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                rationale TEXT,
                risk_snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                approval_state TEXT,
                review_item_id UUID REFERENCES lysara.review_queue(id) ON DELETE SET NULL,
                decided_by TEXT,
                decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                final_status TEXT,
                execution_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                source_ref TEXT,
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.trade_performance (
                metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                trade_id TEXT UNIQUE,
                source_trade_ref TEXT,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy_key TEXT,
                strategy_name TEXT,
                sector TEXT,
                regime_label TEXT,
                entry_price DOUBLE PRECISION,
                exit_price DOUBLE PRECISION,
                quantity DOUBLE PRECISION,
                fees DOUBLE PRECISION,
                pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
                pnl_pct DOUBLE PRECISION,
                win BOOLEAN,
                reconciled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                closed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.signal_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                market TEXT,
                symbol TEXT,
                strategy_key TEXT,
                signal_type TEXT NOT NULL,
                signal_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                strength DOUBLE PRECISION,
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.market_event_memory (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                market TEXT,
                symbol TEXT,
                event_type TEXT,
                headline TEXT,
                impact_level TEXT,
                event_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                source_ref TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.operator_overrides (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                override_type TEXT NOT NULL,
                symbol TEXT,
                strategy_key TEXT,
                market TEXT,
                old_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                new_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                reason TEXT,
                set_by TEXT,
                set_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                cleared_at TIMESTAMPTZ,
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.regime_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                market TEXT NOT NULL,
                symbol TEXT,
                regime_label TEXT NOT NULL,
                volatility_score DOUBLE PRECISION,
                trend_score DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                recommended_params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                applied BOOLEAN NOT NULL DEFAULT FALSE,
                source TEXT NOT NULL DEFAULT 'heartbeat',
                source_ref TEXT,
                observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.open_loops (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                loop_type TEXT NOT NULL DEFAULT 'general',
                title TEXT NOT NULL,
                description TEXT,
                symbol TEXT,
                strategy_key TEXT,
                market TEXT,
                priority DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                trigger_conditions_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'open',
                owner TEXT NOT NULL DEFAULT 'lysara',
                opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                due_hint TEXT,
                closed_at TIMESTAMPTZ,
                closed_reason TEXT,
                source_ref TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.research_notes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                note_hash TEXT NOT NULL UNIQUE,
                note_type TEXT NOT NULL,
                symbol TEXT,
                strategy_key TEXT,
                market TEXT,
                title TEXT,
                content TEXT NOT NULL,
                tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                source_ref TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lysara.market_theses (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                thesis_key TEXT NOT NULL UNIQUE,
                scope_type TEXT NOT NULL,
                symbol TEXT,
                sector TEXT,
                strategy_key TEXT,
                thesis_text TEXT NOT NULL,
                confidence DOUBLE PRECISION,
                status TEXT NOT NULL DEFAULT 'active',
                supporting_refs_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                source_note_id UUID REFERENCES lysara.research_notes(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_lysara_active_positions_state_symbol_market ON lysara.active_positions_state(symbol, market, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_market_working_state_symbol_market ON lysara.market_working_state(symbol, market, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_strategy_runtime_state_strategy_market ON lysara.strategy_runtime_state(strategy_key, market, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_review_queue_status_priority_req ON lysara.review_queue(status, priority, requested_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_open_loops_status_priority_opened ON lysara.open_loops(status, priority, opened_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_trade_decision_log_symbol_market ON lysara.trade_decision_log(symbol, market, decided_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_trade_performance_symbol_market ON lysara.trade_performance(symbol, market, closed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_signal_history_symbol_market ON lysara.signal_history(symbol, market, observed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_regime_history_market_obs ON lysara.regime_history(market, observed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_research_notes_symbol_market ON lysara.research_notes(symbol, market, recorded_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lysara_market_theses_symbol_strategy ON lysara.market_theses(symbol, strategy_key, updated_at DESC)",
        ]
        with pooled_cursor(commit=True) as cur:
            cur.execute("SET statement_timeout = 0")
            for statement in statements + phase2_plus:
                cur.execute(statement)
        return {"ok": True, "schema": "lysara"}

    def _upsert_sync_state(
        self,
        source_name: str,
        *,
        payload_updated_at: Optional[Any] = None,
        stale_after_seconds: Optional[int] = None,
        error: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        stale_after = stale_after_seconds if stale_after_seconds is not None else self.staleness_defaults.get(source_name)
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.sync_state (
                    source_name, last_success_at, payload_updated_at, stale_after_seconds, error, metadata_json
                )
                VALUES (
                    %s,
                    NOW(),
                    COALESCE(%s::timestamptz, NOW()),
                    %s,
                    %s,
                    %s::jsonb
                )
                ON CONFLICT (source_name) DO UPDATE
                SET last_success_at = NOW(),
                    payload_updated_at = COALESCE(EXCLUDED.payload_updated_at, lysara.sync_state.payload_updated_at),
                    stale_after_seconds = COALESCE(EXCLUDED.stale_after_seconds, lysara.sync_state.stale_after_seconds),
                    error = EXCLUDED.error,
                    metadata_json = EXCLUDED.metadata_json
                """,
                (
                    source_name,
                    payload_updated_at,
                    stale_after,
                    error,
                    _json_dumps(metadata or {}),
                ),
            )

    def import_risk_policy_from_markdown(
        self,
        markdown_text: str,
        *,
        actor: str = "system",
        source_ref: str = "RISK.md",
        policy_key: str = "default",
    ) -> Dict[str, Any]:
        policy = self.parse_risk_markdown(markdown_text)
        limits: Dict[str, Any] = {}
        approval_thresholds: Dict[str, Any] = {}
        autonomy_rules: Dict[str, Any] = {}
        for key, value in policy.items():
            if "approval" in key:
                approval_thresholds[key] = value
            elif "autonomous" in key or "override" in key:
                autonomy_rules[key] = value
            else:
                limits[key] = value
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.risk_policies (
                    policy_key, name, scope, limits_json, approval_thresholds_json,
                    autonomy_rules_json, status, effective_at, source_ref, payload_json, updated_at
                )
                VALUES (
                    %s, %s, 'global', %s::jsonb, %s::jsonb,
                    %s::jsonb, 'active', NOW(), %s, %s::jsonb, NOW()
                )
                ON CONFLICT (policy_key) DO UPDATE
                SET name = EXCLUDED.name,
                    limits_json = EXCLUDED.limits_json,
                    approval_thresholds_json = EXCLUDED.approval_thresholds_json,
                    autonomy_rules_json = EXCLUDED.autonomy_rules_json,
                    status = 'active',
                    source_ref = EXCLUDED.source_ref,
                    payload_json = EXCLUDED.payload_json,
                    updated_at = NOW()
                RETURNING id, policy_key, limits_json, approval_thresholds_json, autonomy_rules_json
                """,
                (
                    policy_key,
                    "Default Lysara Risk Policy",
                    _json_dumps(limits),
                    _json_dumps(approval_thresholds),
                    _json_dumps(autonomy_rules),
                    source_ref,
                    _json_dumps(policy),
                ),
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT COALESCE(MAX(revision_no), 0) FROM lysara.risk_policy_revisions WHERE risk_policy_id = %s",
                (row[0],),
            )
            revision_no = int((cur.fetchone() or [0])[0] or 0)
            cur.execute(
                """
                SELECT limits_json, approval_thresholds_json, autonomy_rules_json
                FROM lysara.risk_policy_revisions
                WHERE risk_policy_id = %s
                ORDER BY revision_no DESC
                LIMIT 1
                """,
                (row[0],),
            )
            existing = cur.fetchone()
            changed = (
                existing is None
                or existing[0] != row[2]
                or existing[1] != row[3]
                or existing[2] != row[4]
            )
            if changed:
                cur.execute(
                    """
                    INSERT INTO lysara.risk_policy_revisions (
                        risk_policy_id, revision_no, limits_json, approval_thresholds_json,
                        autonomy_rules_json, change_summary, change_reason, changed_by,
                        approved_by, approved_at, source_ref
                    )
                    VALUES (
                        %s, %s, %s::jsonb, %s::jsonb,
                        %s::jsonb, %s, %s, %s,
                        %s, NOW(), %s
                    )
                    """,
                    (
                        row[0],
                        revision_no + 1,
                        _json_dumps(limits),
                        _json_dumps(approval_thresholds),
                        _json_dumps(autonomy_rules),
                        "Imported from markdown risk policy",
                        "markdown_sync",
                        actor,
                        actor,
                        source_ref,
                    ),
                )
        return {
            "policy_key": policy_key,
            "limits": limits,
            "approval_thresholds": approval_thresholds,
            "autonomy_rules": autonomy_rules,
            "source_ref": source_ref,
        }

    def _ensure_strategy_profile(
        self,
        strategy_key: str,
        *,
        market_scope: Optional[List[str]] = None,
        source_ref: str = "ops.status.strategy_registry",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        strategy_key = str(strategy_key or "").strip()
        if not strategy_key:
            return
        payload = payload or {}
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.strategy_profiles (
                    strategy_key, name, description, market_scope, allowed_symbols_json,
                    default_params_json, status, owner, source_ref, payload_json, updated_at
                )
                VALUES (
                    %s, %s, %s, %s::jsonb, %s::jsonb,
                    %s::jsonb, 'active', %s, %s, %s::jsonb, NOW()
                )
                ON CONFLICT (strategy_key) DO UPDATE
                SET name = EXCLUDED.name,
                    description = COALESCE(NULLIF(EXCLUDED.description, ''), lysara.strategy_profiles.description),
                    market_scope = CASE
                        WHEN lysara.strategy_profiles.market_scope = '[]'::jsonb THEN EXCLUDED.market_scope
                        ELSE lysara.strategy_profiles.market_scope
                    END,
                    default_params_json = CASE
                        WHEN lysara.strategy_profiles.default_params_json = '{}'::jsonb THEN EXCLUDED.default_params_json
                        ELSE lysara.strategy_profiles.default_params_json
                    END,
                    source_ref = EXCLUDED.source_ref,
                    payload_json = EXCLUDED.payload_json,
                    updated_at = NOW()
                """,
                (
                    strategy_key,
                    payload.get("name") or strategy_key,
                    payload.get("description") or "",
                    _json_dumps(market_scope or []),
                    _json_dumps(payload.get("allowed_symbols") or []),
                    _json_dumps(payload.get("params") or payload.get("default_params") or {}),
                    payload.get("owner") or "lysara",
                    source_ref,
                    _json_dumps(payload),
                ),
            )

    def mirror_status_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = payload or {}
        active_markets = [str(item).strip().lower() for item in (payload.get("active_markets") or []) if str(item).strip()]
        total_equity = None
        equity = payload.get("equity")
        if isinstance(equity, dict):
            total_equity = round(sum(float(v or 0.0) for v in equity.values()), 4)
        simulation = payload.get("simulation_portfolio") or {}
        runtime_flags = {
            "feed_freshness": payload.get("feed_freshness") or {},
            "broker_health": payload.get("broker_health") or {},
            "risk_managers": payload.get("risk_managers") or {},
            "symbol_controls": payload.get("symbol_controls") or {},
            "strategy_controls": payload.get("strategy_controls") or {},
            "last_heartbeat_at": payload.get("last_heartbeat_at"),
        }
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.portfolio_working_state (
                    portfolio_key, market, as_of, paused, pause_reason, simulation_mode,
                    live_trading_enabled, autonomous_mode, total_equity, cash, buying_power,
                    portfolio_value, runtime_flags_json, payload_json, source_ref, updated_at
                )
                VALUES (
                    'default', 'all', COALESCE(%s::timestamptz, NOW()), %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s, NOW()
                )
                ON CONFLICT (portfolio_key, market) DO UPDATE
                SET as_of = EXCLUDED.as_of,
                    paused = EXCLUDED.paused,
                    pause_reason = EXCLUDED.pause_reason,
                    simulation_mode = EXCLUDED.simulation_mode,
                    live_trading_enabled = EXCLUDED.live_trading_enabled,
                    autonomous_mode = EXCLUDED.autonomous_mode,
                    total_equity = EXCLUDED.total_equity,
                    cash = EXCLUDED.cash,
                    buying_power = EXCLUDED.buying_power,
                    portfolio_value = EXCLUDED.portfolio_value,
                    runtime_flags_json = EXCLUDED.runtime_flags_json,
                    payload_json = EXCLUDED.payload_json,
                    source_ref = EXCLUDED.source_ref,
                    updated_at = NOW()
                """,
                (
                    payload.get("updated_at") or payload.get("timestamp"),
                    bool(payload.get("paused")),
                    payload.get("pause_reason") or "",
                    bool(payload.get("simulation_mode")),
                    bool(payload.get("live_trading_enabled")),
                    bool(payload.get("autonomous_mode")),
                    total_equity,
                    _safe_float(simulation.get("cash")),
                    _safe_float(simulation.get("buying_power")),
                    _safe_float(simulation.get("portfolio_value")) or total_equity,
                    _json_dumps(runtime_flags),
                    _json_dumps(payload),
                    "ops.get_status",
                ),
            )
            for market in active_markets or ["all"]:
                cur.execute(
                    """
                    INSERT INTO lysara.current_regime_state (
                        market, regime_label, observed_at, source_ref, payload_json, updated_at
                    )
                    VALUES (
                        %s, %s, COALESCE(%s::timestamptz, NOW()), %s, %s::jsonb, NOW()
                    )
                    ON CONFLICT (market) DO UPDATE
                    SET regime_label = EXCLUDED.regime_label,
                        observed_at = EXCLUDED.observed_at,
                        source_ref = EXCLUDED.source_ref,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = NOW()
                    """,
                    (
                        market,
                        payload.get("regime"),
                        payload.get("updated_at") or payload.get("timestamp"),
                        "ops.get_status",
                        _json_dumps({"status_payload": payload}),
                    ),
                )
            registry = payload.get("strategy_registry") or {}
            controls = payload.get("strategy_controls") or {}
            for market, items in registry.items() if isinstance(registry, dict) else []:
                strategies: List[str] = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            name = str(item.get("strategy_name") or item.get("name") or "").strip()
                            if name:
                                strategies.append(name)
                                self._ensure_strategy_profile(name, market_scope=[str(market).lower()], payload=item)
                        else:
                            name = str(item).strip()
                            if name:
                                strategies.append(name)
                                self._ensure_strategy_profile(name, market_scope=[str(market).lower()])
                for strategy_name in strategies:
                    enabled = bool((controls.get(market) or {}).get(strategy_name, True))
                    cur.execute(
                        """
                        INSERT INTO lysara.strategy_runtime_state (
                            strategy_key, market, status, paused, pause_reason, runtime_params_json,
                            autonomy_mode, symbol_controls_json, source_ref, payload_json, updated_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s::jsonb,
                            %s, %s::jsonb, %s, %s::jsonb, NOW()
                        )
                        ON CONFLICT (strategy_key, market) DO UPDATE
                        SET status = EXCLUDED.status,
                            paused = EXCLUDED.paused,
                            pause_reason = EXCLUDED.pause_reason,
                            runtime_params_json = EXCLUDED.runtime_params_json,
                            autonomy_mode = EXCLUDED.autonomy_mode,
                            symbol_controls_json = EXCLUDED.symbol_controls_json,
                            source_ref = EXCLUDED.source_ref,
                            payload_json = EXCLUDED.payload_json,
                            updated_at = NOW()
                        """,
                        (
                            strategy_name,
                            str(market).lower(),
                            "active" if enabled else "disabled",
                            not enabled,
                            "" if enabled else "remote_control_disabled",
                            _json_dumps({}),
                            "autonomous" if bool(payload.get("autonomous_mode")) else "manual",
                            _json_dumps((payload.get("symbol_controls") or {}).get(market) or {}),
                            "ops.get_status",
                            _json_dumps({"registry": items, "controls": controls}),
                        ),
                    )
        self._upsert_sync_state(
            "status",
            payload_updated_at=payload.get("updated_at") or payload.get("timestamp"),
            metadata={"active_markets": active_markets},
        )
        return {"ok": True, "active_markets": active_markets}

    def mirror_portfolio_payload(self, payload: Dict[str, Any], market: str = "all") -> Dict[str, Any]:
        payload = payload or {}
        market = (market or payload.get("market") or "all").strip().lower()
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.portfolio_working_state (
                    portfolio_key, market, as_of, total_equity, cash, buying_power,
                    portfolio_value, payload_json, source_ref, updated_at
                )
                VALUES (
                    'default', %s, COALESCE(%s::timestamptz, NOW()), %s, %s, %s, %s,
                    %s::jsonb, %s, NOW()
                )
                ON CONFLICT (portfolio_key, market) DO UPDATE
                SET as_of = EXCLUDED.as_of,
                    total_equity = EXCLUDED.total_equity,
                    cash = EXCLUDED.cash,
                    buying_power = EXCLUDED.buying_power,
                    portfolio_value = EXCLUDED.portfolio_value,
                    payload_json = EXCLUDED.payload_json,
                    source_ref = EXCLUDED.source_ref,
                    updated_at = NOW()
                """,
                (
                    market,
                    payload.get("updated_at") or payload.get("timestamp"),
                    _safe_float(payload.get("total_equity")),
                    _safe_float(payload.get("cash")),
                    _safe_float(payload.get("buying_power")),
                    _safe_float(payload.get("portfolio_value") or payload.get("total_equity")),
                    _json_dumps(payload),
                    "ops.get_portfolio",
                ),
            )
        self._upsert_sync_state(
            "portfolio",
            payload_updated_at=payload.get("updated_at") or payload.get("timestamp"),
            metadata={"market": market},
        )
        return {"ok": True, "market": market}

    def mirror_positions_payload(self, payload: Dict[str, Any], market: Optional[str] = None) -> Dict[str, Any]:
        payload = payload or {}
        account_scope = str(payload.get("account_scope") or payload.get("account") or "default")
        items = payload.get("items") or payload.get("positions") or []
        incoming_by_market: Dict[str, List[str]] = {}
        with pooled_cursor(commit=True) as cur:
            for row in items:
                row = row or {}
                row_market = str(row.get("market") or market or payload.get("market") or "all").strip().lower()
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                incoming_by_market.setdefault(row_market, []).append(symbol)
                cur.execute(
                    """
                    INSERT INTO lysara.active_positions_state (
                        account_scope, market, symbol, side, quantity, entry_price, mark_price,
                        unrealized_pnl, strategy_key, stop_json, target_json, risk_state_json,
                        status, as_of, payload_json, source_ref, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
                        %s, COALESCE(%s::timestamptz, NOW()), %s::jsonb, %s, NOW()
                    )
                    ON CONFLICT (account_scope, market, symbol) DO UPDATE
                    SET side = EXCLUDED.side,
                        quantity = EXCLUDED.quantity,
                        entry_price = EXCLUDED.entry_price,
                        mark_price = EXCLUDED.mark_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        strategy_key = EXCLUDED.strategy_key,
                        stop_json = EXCLUDED.stop_json,
                        target_json = EXCLUDED.target_json,
                        risk_state_json = EXCLUDED.risk_state_json,
                        status = EXCLUDED.status,
                        as_of = EXCLUDED.as_of,
                        payload_json = EXCLUDED.payload_json,
                        source_ref = EXCLUDED.source_ref,
                        updated_at = NOW()
                    """,
                    (
                        account_scope,
                        row_market,
                        symbol,
                        row.get("side"),
                        _safe_float(row.get("quantity") or row.get("qty") or row.get("size")),
                        _safe_float(row.get("entry_price") or row.get("avg_entry_price")),
                        _safe_float(row.get("mark_price") or row.get("current_price") or row.get("price")),
                        _safe_float(row.get("unrealized_pnl") or row.get("pnl")),
                        row.get("strategy_key") or row.get("strategy_name"),
                        _json_dumps(row.get("stop") or row.get("stop_loss") or {}),
                        _json_dumps(row.get("target") or row.get("take_profit") or {}),
                        _json_dumps(
                            {
                                "risk_pct": row.get("risk_pct"),
                                "effective_weight_pct": row.get("effective_weight_pct"),
                                "heat_pct": row.get("heat_pct"),
                            }
                        ),
                        row.get("status") or "open",
                        row.get("updated_at") or row.get("timestamp") or payload.get("updated_at"),
                        _json_dumps(row),
                        "ops.get_positions",
                    ),
                )
            for row_market, symbols in incoming_by_market.items():
                cur.execute(
                    """
                    DELETE FROM lysara.active_positions_state
                    WHERE account_scope = %s
                      AND market = %s
                      AND symbol <> ALL(%s)
                    """,
                    (account_scope, row_market, symbols),
                )
        self._upsert_sync_state(
            "positions",
            payload_updated_at=payload.get("updated_at"),
            metadata={"account_scope": account_scope, "markets": list(incoming_by_market.keys())},
        )
        return {"ok": True, "position_count": len(items)}

    def _mirror_market_rows(
        self,
        *,
        source_name: str,
        payload: Dict[str, Any],
        market: Optional[str] = None,
        rows: Sequence[Dict[str, Any]],
        extractor,
    ) -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            for row in rows:
                symbol, top_signals, catalysts, risk_posture, confidence, metadata = extractor(row)
                cur.execute(
                    """
                    INSERT INTO lysara.market_working_state (
                        source_name, market, symbol, as_of, top_signals_json, catalysts_json,
                        risk_posture, feed_freshness, confidence, metadata_json, payload_json, updated_at
                    )
                    VALUES (
                        %s, %s, %s, COALESCE(%s::timestamptz, NOW()), %s::jsonb, %s::jsonb,
                        %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, NOW()
                    )
                    ON CONFLICT (source_name, market, symbol) DO UPDATE
                    SET as_of = EXCLUDED.as_of,
                        top_signals_json = EXCLUDED.top_signals_json,
                        catalysts_json = EXCLUDED.catalysts_json,
                        risk_posture = EXCLUDED.risk_posture,
                        feed_freshness = EXCLUDED.feed_freshness,
                        confidence = EXCLUDED.confidence,
                        metadata_json = EXCLUDED.metadata_json,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = NOW()
                    """,
                    (
                        source_name,
                        str(market or payload.get("market") or "all").strip().lower(),
                        symbol,
                        payload.get("updated_at") or payload.get("timestamp"),
                        _json_dumps(top_signals),
                        _json_dumps(catalysts),
                        risk_posture,
                        _json_dumps(payload.get("feed_freshness") or {}),
                        confidence,
                        _json_dumps(metadata),
                        _json_dumps(row),
                    ),
                )
        self._upsert_sync_state(
            source_name,
            payload_updated_at=payload.get("updated_at") or payload.get("timestamp"),
            metadata={"row_count": len(rows)},
        )
        return {"ok": True, "source_name": source_name, "row_count": len(rows)}

    def mirror_market_snapshot_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prices = payload.get("prices") or {}
        rows = [{"symbol": symbol, **(raw if isinstance(raw, dict) else {"value": raw})} for symbol, raw in prices.items()]

        def extract(row: Dict[str, Any]):
            symbol = str(row.get("symbol") or "*").upper()
            top_signals = [{"name": "change_pct_24h", "value": row.get("change_pct_24h")}]
            confidence = _safe_float(row.get("trend_score")) or _safe_float(row.get("confidence"))
            risk_posture = "volatile" if _safe_float(row.get("change_pct_24h")) and abs(float(row.get("change_pct_24h") or 0.0)) >= 4 else "neutral"
            return symbol, top_signals, [], risk_posture, confidence, {"price": row.get("price"), "trend_score": row.get("trend_score")}

        return self._mirror_market_rows(
            source_name="market_snapshot",
            payload=payload,
            market=payload.get("market"),
            rows=rows,
            extractor=extract,
        )

    def mirror_sentiment_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def extract(row: Dict[str, Any]):
            symbol = str(row.get("symbol") or "*").upper()
            risk_posture = "bullish" if float(row.get("score") or 0.0) > 0 else "bearish" if float(row.get("score") or 0.0) < 0 else "mixed"
            return symbol, [{"name": "sentiment_score", "value": row.get("score")}], row.get("anomaly_flags") or [], risk_posture, _safe_float(row.get("confidence")), {"mention_velocity": row.get("mention_velocity")}

        return self._mirror_market_rows(
            source_name="sentiment",
            payload=payload,
            rows=payload.get("symbols") or [],
            extractor=extract,
        )

    def mirror_confluence_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def extract(row: Dict[str, Any]):
            symbol = str(row.get("symbol") or "*").upper()
            catalysts = [{"support": row.get("support")}, {"resistance": row.get("resistance")}]
            return symbol, [{"name": "confluence_score", "value": row.get("confluence_score")}], catalysts, str(row.get("alignment_label") or "mixed"), _safe_float(row.get("confidence")), {"breakout_probability": row.get("breakout_probability")}

        return self._mirror_market_rows(
            source_name="confluence",
            payload=payload,
            rows=payload.get("symbols") or [],
            extractor=extract,
        )

    def mirror_event_risk_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = payload.get("symbols") or []
        with pooled_cursor(commit=True) as cur:
            for event in payload.get("events") or []:
                cur.execute(
                    """
                    INSERT INTO lysara.market_event_memory (
                        event_at, market, symbol, event_type, headline, impact_level, event_payload_json, source_ref
                    )
                    VALUES (
                        COALESCE(%s::timestamptz, NOW()), %s, %s, %s, %s, %s, %s::jsonb, %s
                    )
                    """,
                    (
                        event.get("starts_at"),
                        event.get("scope") or payload.get("market"),
                        (event.get("symbols") or [None])[0],
                        event.get("category"),
                        event.get("title"),
                        event.get("severity"),
                        _json_dumps(event),
                        "ops.get_event_risk",
                    ),
                )

        def extract(row: Dict[str, Any]):
            symbol = str(row.get("symbol") or "*").upper()
            return symbol, [{"name": "risk_score", "value": row.get("risk_score")}], row.get("upcoming_events") or [], str(row.get("action") or "watch"), _safe_float(row.get("risk_score")), {"block_new_positions": row.get("block_new_positions")}

        return self._mirror_market_rows(
            source_name="event_risk",
            payload=payload,
            rows=rows,
            extractor=extract,
        )

    def mirror_exposure_payload(self, payload: Dict[str, Any], market: str = "crypto") -> Dict[str, Any]:
        market = str(market or payload.get("market") or "crypto").lower()
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.portfolio_working_state (
                    portfolio_key, market, as_of, portfolio_value, gross_exposure_pct,
                    heat_score, payload_json, source_ref, updated_at
                )
                VALUES (
                    'default', %s, COALESCE(%s::timestamptz, NOW()), %s, %s, %s,
                    %s::jsonb, %s, NOW()
                )
                ON CONFLICT (portfolio_key, market) DO UPDATE
                SET as_of = EXCLUDED.as_of,
                    portfolio_value = COALESCE(EXCLUDED.portfolio_value, lysara.portfolio_working_state.portfolio_value),
                    gross_exposure_pct = EXCLUDED.gross_exposure_pct,
                    heat_score = EXCLUDED.heat_score,
                    payload_json = EXCLUDED.payload_json,
                    source_ref = EXCLUDED.source_ref,
                    updated_at = NOW()
                """,
                (
                    market,
                    payload.get("updated_at") or payload.get("timestamp"),
                    _safe_float(payload.get("portfolio_value")),
                    _safe_float(payload.get("gross_exposure_pct")),
                    _safe_float(payload.get("heat_score") or payload.get("total_effective_heat_pct")),
                    _json_dumps(payload),
                    "ops.get_exposure",
                ),
            )

        def extract(row: Dict[str, Any]):
            symbol = str(row.get("symbol") or "*").upper()
            posture = "concentrated" if float(row.get("effective_weight_pct") or 0.0) >= 15 else "normal"
            return symbol, [{"name": "effective_weight_pct", "value": row.get("effective_weight_pct")}], [], posture, _safe_float(row.get("heat_pct") or row.get("effective_weight_pct")), {"sector": row.get("sector")}

        return self._mirror_market_rows(
            source_name="exposure",
            payload={"market": market, **(payload or {})},
            market=market,
            rows=payload.get("positions") or [],
            extractor=extract,
        )

    def mirror_override_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE lysara.portfolio_working_state
                SET runtime_flags_json = COALESCE(runtime_flags_json, '{}'::jsonb) || %s::jsonb,
                    updated_at = NOW()
                WHERE portfolio_key = 'default' AND market = 'all'
                """,
                (_json_dumps({"override": payload}),),
            )
        self._upsert_sync_state(
            "status",
            payload_updated_at=payload.get("activated_at") or payload.get("expires_at"),
            metadata={"override_enabled": bool(payload.get("enabled"))},
        )
        return {"ok": True, "enabled": bool(payload.get("enabled"))}

    def mirror_incidents_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        items = payload.get("items") or []
        unresolved = 0
        with pooled_cursor(commit=True) as cur:
            for row in items:
                incident_id = str(row.get("incident_id") or row.get("id") or "").strip()
                status = str(row.get("status") or "pending").strip().lower()
                source_ref = f"incident:{incident_id}" if incident_id else f"incident:{hashlib.sha1(_json_dumps(row).encode('utf-8')).hexdigest()[:16]}"
                if status in {"resolved", "closed"}:
                    cur.execute(
                        """
                        UPDATE lysara.review_queue
                        SET status = 'resolved', resolved_at = NOW(), resolution_note = %s, payload_json = %s::jsonb
                        WHERE queue_type = 'incident' AND source_ref = %s
                        """,
                        (row.get("resolution") or status, _json_dumps(row), source_ref),
                    )
                    continue
                unresolved += 1
                cur.execute(
                    """
                    INSERT INTO lysara.review_queue (
                        queue_type, symbol, strategy_key, market, priority, title, details_json,
                        status, requested_at, source_ref, external_ref, payload_json
                    )
                    VALUES (
                        'incident', %s, %s, %s, %s, %s, %s::jsonb,
                        'pending', COALESCE(%s::timestamptz, NOW()), %s, %s, %s::jsonb
                    )
                    ON CONFLICT (queue_type, source_ref) DO UPDATE
                    SET symbol = EXCLUDED.symbol,
                        strategy_key = EXCLUDED.strategy_key,
                        market = EXCLUDED.market,
                        priority = EXCLUDED.priority,
                        title = EXCLUDED.title,
                        details_json = EXCLUDED.details_json,
                        status = 'pending',
                        payload_json = EXCLUDED.payload_json
                    """,
                    (
                        row.get("symbol"),
                        row.get("strategy_key"),
                        row.get("market"),
                        _safe_float(row.get("priority")) or 0.8,
                        row.get("title") or row.get("summary") or "Lysara incident",
                        _json_dumps(row),
                        row.get("created_at") or row.get("opened_at") or row.get("timestamp"),
                        source_ref,
                        incident_id or None,
                        _json_dumps(row),
                    ),
                )
        self._upsert_sync_state(
            "status",
            payload_updated_at=payload.get("updated_at"),
            metadata={"incident_count": len(items), "unresolved": unresolved},
        )
        return {"ok": True, "incident_count": len(items), "unresolved": unresolved}

    def _research_note_hash(self, note_type: str, payload: Dict[str, Any]) -> str:
        seed = "|".join(
            [
                note_type,
                str(payload.get("market") or ""),
                str(payload.get("symbol") or ""),
                str(payload.get("strategy_key") or payload.get("strategy_name") or ""),
                str(payload.get("title") or ""),
                str(payload.get("summary") or payload.get("content") or ""),
                str(payload.get("recorded_at") or payload.get("created_at") or payload.get("updated_at") or ""),
            ]
        )
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()

    def mirror_research_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = payload.get("items") or payload.get("research") or [payload]
        stored = 0
        with pooled_cursor(commit=True) as cur:
            for row in rows:
                note_hash = self._research_note_hash("research", row or {})
                cur.execute(
                    """
                    INSERT INTO lysara.research_notes (
                        note_hash, note_type, symbol, strategy_key, market, title, content,
                        tags_json, payload_json, recorded_at, source_ref
                    )
                    VALUES (
                        %s, 'research', %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb, COALESCE(%s::timestamptz, NOW()), %s
                    )
                    ON CONFLICT (note_hash) DO NOTHING
                    """,
                    (
                        note_hash,
                        row.get("symbol"),
                        row.get("strategy_key") or row.get("strategy_name"),
                        row.get("market"),
                        row.get("title") or row.get("headline") or row.get("symbol"),
                        row.get("summary") or row.get("content") or "",
                        _json_dumps(row.get("tags") or row.get("bullish_factors") or row.get("bearish_factors") or []),
                        _json_dumps(row),
                        row.get("recorded_at") or row.get("created_at") or row.get("updated_at") or row.get("stale_after"),
                        "ops.research",
                    ),
                )
                stored += cur.rowcount
        self._upsert_sync_state("research", payload_updated_at=payload.get("updated_at"), metadata={"stored": stored})
        return {"ok": True, "stored": stored}

    def mirror_journal_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = payload.get("items") or payload.get("journal") or [payload]
        stored = 0
        with pooled_cursor(commit=True) as cur:
            for row in rows:
                note_hash = self._research_note_hash("journal", row or {})
                cur.execute(
                    """
                    INSERT INTO lysara.research_notes (
                        note_hash, note_type, symbol, strategy_key, market, title, content,
                        tags_json, payload_json, recorded_at, source_ref
                    )
                    VALUES (
                        %s, 'journal', %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb, COALESCE(%s::timestamptz, NOW()), %s
                    )
                    ON CONFLICT (note_hash) DO NOTHING
                    """,
                    (
                        note_hash,
                        row.get("symbol"),
                        row.get("strategy_key") or row.get("strategy_name"),
                        row.get("market"),
                        row.get("action") or row.get("title") or "journal",
                        row.get("summary") or row.get("content") or "",
                        _json_dumps([row.get("mode")] if row.get("mode") else []),
                        _json_dumps(row),
                        row.get("recorded_at") or row.get("created_at") or row.get("updated_at"),
                        "ops.journal",
                    ),
                )
                stored += cur.rowcount
        self._upsert_sync_state("journal", payload_updated_at=payload.get("updated_at"), metadata={"stored": stored})
        return {"ok": True, "stored": stored}

    def record_trade_decision(
        self,
        *,
        trade_payload: Dict[str, Any],
        risk_snapshot: Optional[Dict[str, Any]] = None,
        decision_type: str = "trade_intent",
        rationale: str = "",
        approval_state: str = "pending",
        decided_by: str = "operator",
        source_ref: Optional[str] = None,
        review_item_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        final_status: Optional[str] = None,
        execution_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if decision_id:
            with pooled_cursor(commit=True) as cur:
                cur.execute(
                    """
                    UPDATE lysara.trade_decision_log
                    SET approval_state = COALESCE(%s, approval_state),
                        final_status = COALESCE(%s, final_status),
                        execution_payload_json = COALESCE(%s::jsonb, execution_payload_json),
                        metadata_json = COALESCE(metadata_json, '{}'::jsonb) || %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s::uuid
                    RETURNING id, approval_state, final_status
                    """,
                    (
                        approval_state,
                        final_status,
                        _json_dumps(execution_payload or {}) if execution_payload is not None else None,
                        _json_dumps(metadata or {}),
                        decision_id,
                    ),
                )
                row = cur.fetchone()
            return {"decision_id": str(row[0]) if row else decision_id, "approval_state": row[1] if row else approval_state, "final_status": row[2] if row else final_status}

        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.trade_decision_log (
                    trade_ref, decision_type, symbol, strategy_key, market, regime_label,
                    signal_snapshot_json, rationale, risk_snapshot_json, approval_state,
                    review_item_id, decided_by, decided_at, final_status, execution_payload_json,
                    source_ref, metadata_json
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s::jsonb, %s,
                    %s::uuid, %s, NOW(), %s, %s::jsonb,
                    %s, %s::jsonb
                )
                RETURNING id, approval_state
                """,
                (
                    trade_payload.get("dedupe_nonce") or trade_payload.get("trade_id"),
                    decision_type,
                    trade_payload.get("symbol"),
                    trade_payload.get("strategy_key") or trade_payload.get("strategy_name"),
                    trade_payload.get("market"),
                    trade_payload.get("regime_label"),
                    _json_dumps(trade_payload.get("signal_snapshot") or {}),
                    rationale or trade_payload.get("thesis") or "",
                    _json_dumps(risk_snapshot or {}),
                    approval_state,
                    review_item_id,
                    decided_by,
                    final_status,
                    _json_dumps(execution_payload or {}),
                    source_ref,
                    _json_dumps(metadata or {}),
                ),
            )
            row = cur.fetchone()
        return {"decision_id": str(row[0]), "approval_state": row[1]}

    def record_operator_override(
        self,
        *,
        override_type: str,
        actor: str,
        reason: str = "",
        market: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_key: Optional[str] = None,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        source_ref: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.operator_overrides (
                    override_type, symbol, strategy_key, market, old_value_json, new_value_json,
                    reason, set_by, set_at, source_ref, payload_json
                )
                VALUES (
                    %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                    %s, %s, NOW(), %s, %s::jsonb
                )
                RETURNING id, set_at
                """,
                (
                    override_type,
                    symbol,
                    strategy_key,
                    market,
                    _json_dumps(old_value or {}),
                    _json_dumps(new_value or {}),
                    reason,
                    actor,
                    source_ref,
                    _json_dumps(payload or {}),
                ),
            )
            row = cur.fetchone()
        return {"override_id": str(row[0]), "set_at": row[1].isoformat() if row and row[1] else None}

    def create_review_item_from_trade_intent(
        self,
        *,
        trade_payload: Dict[str, Any],
        risk_snapshot: Dict[str, Any],
        proactive_note: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_ref = str((proactive_note or {}).get("note_id") or f"trade-approval:{hashlib.sha1(_json_dumps(trade_payload).encode('utf-8')).hexdigest()[:16]}")
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.review_queue (
                    queue_type, symbol, strategy_key, market, priority, title, details_json,
                    status, requested_at, source_ref, external_ref, payload_json
                )
                VALUES (
                    'trade_approval', %s, %s, %s, %s, %s, %s::jsonb,
                    'pending', NOW(), %s, %s, %s::jsonb
                )
                ON CONFLICT (queue_type, source_ref) DO UPDATE
                SET symbol = EXCLUDED.symbol,
                    strategy_key = EXCLUDED.strategy_key,
                    market = EXCLUDED.market,
                    priority = EXCLUDED.priority,
                    title = EXCLUDED.title,
                    details_json = EXCLUDED.details_json,
                    status = 'pending',
                    payload_json = EXCLUDED.payload_json
                RETURNING id, title, status
                """,
                (
                    trade_payload.get("symbol"),
                    trade_payload.get("strategy_key") or trade_payload.get("strategy_name"),
                    trade_payload.get("market"),
                    0.9,
                    f"Trade approval required for {trade_payload.get('symbol')} {trade_payload.get('side')}",
                    _json_dumps({"trade_payload": trade_payload, "risk_snapshot": risk_snapshot, "proactive_note": proactive_note or {}}),
                    source_ref,
                    str((proactive_note or {}).get("note_id") or ""),
                    _json_dumps({"trade_payload": trade_payload, "risk_snapshot": risk_snapshot}),
                ),
            )
            row = cur.fetchone()
        return {"review_item_id": str(row[0]), "title": row[1], "status": row[2], "source_ref": source_ref}

    def create_open_loop(
        self,
        *,
        title: str,
        description: str = "",
        loop_type: str = "general",
        symbol: Optional[str] = None,
        strategy_key: Optional[str] = None,
        market: Optional[str] = None,
        priority: float = 0.5,
        due_hint: str = "",
        trigger_conditions: Optional[Dict[str, Any]] = None,
        source_ref: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.open_loops (
                    loop_type, title, description, symbol, strategy_key, market, priority,
                    trigger_conditions_json, status, owner, opened_at, due_hint, source_ref, payload_json
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, 'open', 'lysara', NOW(), %s, %s, %s::jsonb
                )
                RETURNING id, opened_at
                """,
                (
                    loop_type,
                    title,
                    description,
                    symbol,
                    strategy_key,
                    market,
                    priority,
                    _json_dumps(trigger_conditions or {}),
                    due_hint,
                    source_ref,
                    _json_dumps(payload or {}),
                ),
            )
            row = cur.fetchone()
        return {"loop_id": str(row[0]), "opened_at": row[1].isoformat() if row and row[1] else None}

    def close_open_loop(self, *, loop_id: str, reason: str = "") -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE lysara.open_loops
                SET status = 'closed', closed_at = NOW(), closed_reason = %s
                WHERE id = %s::uuid
                RETURNING id, closed_at
                """,
                (reason, loop_id),
            )
            row = cur.fetchone()
        return {"loop_id": str(row[0]) if row else loop_id, "closed_at": row[1].isoformat() if row and row[1] else None}

    def resolve_review_item(self, item_id: str, *, resolution_note: str = "", status: str = "resolved") -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE lysara.review_queue
                SET status = %s, resolved_at = NOW(), resolution_note = %s
                WHERE id = %s::uuid
                RETURNING id, queue_type
                """,
                (status, resolution_note, item_id),
            )
            row = cur.fetchone()
            if row and row[1] == "trade_approval":
                cur.execute(
                    "UPDATE lysara.trade_decision_log SET approval_state = %s, updated_at = NOW() WHERE review_item_id = %s::uuid",
                    (status, item_id),
                )
        return {"review_item_id": item_id, "status": status}

    def promote_research_note_to_thesis(
        self,
        *,
        note_id: str,
        thesis_key: str,
        confidence: float,
        scope_type: str,
    ) -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                "SELECT symbol, strategy_key, content FROM lysara.research_notes WHERE id = %s::uuid",
                (note_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("research_note_not_found")
            cur.execute(
                """
                INSERT INTO lysara.market_theses (
                    thesis_key, scope_type, symbol, strategy_key, thesis_text, confidence,
                    status, supporting_refs_json, source_note_id, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    'active', %s::jsonb, %s::uuid, NOW()
                )
                ON CONFLICT (thesis_key) DO UPDATE
                SET thesis_text = EXCLUDED.thesis_text,
                    confidence = EXCLUDED.confidence,
                    supporting_refs_json = EXCLUDED.supporting_refs_json,
                    source_note_id = EXCLUDED.source_note_id,
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    thesis_key,
                    scope_type,
                    row[0],
                    row[1],
                    row[2],
                    confidence,
                    _json_dumps([note_id]),
                    note_id,
                ),
            )
            thesis_row = cur.fetchone()
        return {"thesis_id": str(thesis_row[0]), "thesis_key": thesis_key}

    def promote_decision_pattern_to_symbol_profile(self, *, symbol: str, note: str, market: str = "all") -> Dict[str, Any]:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            raise ValueError("symbol_required")
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.symbol_profiles (
                    symbol, market, notes, updated_at
                )
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (symbol, market) DO UPDATE
                SET notes = CASE
                    WHEN COALESCE(lysara.symbol_profiles.notes, '') = '' THEN EXCLUDED.notes
                    ELSE lysara.symbol_profiles.notes || E'\\n' || EXCLUDED.notes
                END,
                    updated_at = NOW()
                RETURNING id
                """,
                (symbol, market, note),
            )
            row = cur.fetchone()
        return {"symbol_profile_id": str(row[0]), "symbol": symbol}

    def backfill_legacy_lysara_data(self) -> Dict[str, Any]:
        with pooled_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO lysara.trade_performance (
                    trade_id, source_trade_ref, market, symbol, strategy_key, strategy_name,
                    sector, regime_label, entry_price, exit_price, quantity, fees, pnl, pnl_pct,
                    win, reconciled_at, closed_at, metadata_json, created_at
                )
                SELECT
                    trade_id,
                    trade_id,
                    market,
                    symbol,
                    strategy_name,
                    strategy_name,
                    sector,
                    regime_label,
                    entry_price,
                    exit_price,
                    quantity,
                    fees,
                    pnl,
                    pnl_pct,
                    win,
                    reconciled_at,
                    closed_at,
                    metadata,
                    created_at
                FROM public.lysara_trade_performance
                ON CONFLICT (trade_id) DO UPDATE
                SET market = EXCLUDED.market,
                    symbol = EXCLUDED.symbol,
                    strategy_key = EXCLUDED.strategy_key,
                    strategy_name = EXCLUDED.strategy_name,
                    sector = EXCLUDED.sector,
                    regime_label = EXCLUDED.regime_label,
                    entry_price = EXCLUDED.entry_price,
                    exit_price = EXCLUDED.exit_price,
                    quantity = EXCLUDED.quantity,
                    fees = EXCLUDED.fees,
                    pnl = EXCLUDED.pnl,
                    pnl_pct = EXCLUDED.pnl_pct,
                    win = EXCLUDED.win,
                    reconciled_at = EXCLUDED.reconciled_at,
                    closed_at = EXCLUDED.closed_at,
                    metadata_json = EXCLUDED.metadata_json
                """
            )
            trade_count = cur.rowcount
            cur.execute(
                """
                INSERT INTO lysara.regime_history (
                    market, symbol, regime_label, volatility_score, trend_score, confidence,
                    recommended_params_json, applied, source, source_ref, observed_at, payload_json
                )
                SELECT
                    market,
                    NULL,
                    regime_label,
                    volatility_score,
                    trend_score,
                    confidence,
                    recommended_params,
                    applied,
                    source,
                    'legacy.public.lysara_market_regimes',
                    observed_at,
                    jsonb_build_object(
                        'market', market,
                        'regime_label', regime_label,
                        'observed_at', observed_at
                    )
                FROM public.lysara_market_regimes src
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM lysara.regime_history dst
                    WHERE dst.market = src.market
                      AND dst.regime_label = src.regime_label
                      AND dst.observed_at = src.observed_at
                )
                """
            )
            regime_count = cur.rowcount
            cur.execute(
                """
                INSERT INTO lysara.review_queue (
                    queue_type, symbol, market, priority, title, details_json, status, requested_at,
                    source_ref, external_ref, payload_json
                )
                SELECT
                    'trade_approval',
                    metadata->'trade_payload'->>'symbol',
                    metadata->'trade_payload'->>'market',
                    0.9,
                    title,
                    metadata,
                    CASE
                        WHEN approval_status = 'approved' THEN 'resolved'
                        WHEN status IN ('completed', 'resolved') THEN 'resolved'
                        ELSE 'pending'
                    END,
                    created_at,
                    note_id::text,
                    note_id::text,
                    metadata
                FROM public.proactive_notes
                WHERE source = 'lysara_trade_intent'
                ON CONFLICT (queue_type, source_ref) DO NOTHING
                """
            )
            note_count = cur.rowcount
        return {
            "ok": True,
            "trade_performance_backfilled": trade_count,
            "regimes_backfilled": regime_count,
            "review_items_backfilled": note_count,
        }

    def get_canonical_risk(self) -> Dict[str, Any]:
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT policy_key, name, scope, limits_json, approval_thresholds_json,
                       autonomy_rules_json, status, effective_at, source_ref, updated_at
                FROM lysara.risk_policies
                WHERE status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
            rows = _rows_to_dicts(cur)
        return {"risk_policies": rows}

    def get_canonical_strategies(self, limit: int = 50) -> Dict[str, Any]:
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT strategy_key, name, description, market_scope, allowed_symbols_json,
                       default_params_json, status, owner, source_ref, updated_at
                FROM lysara.strategy_profiles
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (max(1, min(int(limit or 50), 200)),),
            )
            rows = _rows_to_dicts(cur)
        return {"strategies": rows}

    def get_working_state(
        self,
        *,
        symbol: Optional[str] = None,
        market: Optional[str] = None,
        strategy_key: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit or 20), 100))
        symbol = (symbol or "").strip().upper() or None
        market = (market or "").strip().lower() or None
        strategy_key = (strategy_key or "").strip() or None
        result: Dict[str, Any] = {}
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT portfolio_key, market, as_of, paused, pause_reason, simulation_mode,
                       live_trading_enabled, autonomous_mode, total_equity, cash, buying_power,
                       portfolio_value, gross_exposure_pct, heat_score, runtime_flags_json, source_ref, updated_at
                FROM lysara.portfolio_working_state
                ORDER BY updated_at DESC
                LIMIT 5
                """
            )
            result["portfolio"] = _rows_to_dicts(cur)

            where = ["status IN ('open', 'active')"]
            params: List[Any] = []
            if symbol:
                where.append("symbol = %s")
                params.append(symbol)
            if market:
                where.append("market = %s")
                params.append(market)
            if strategy_key:
                where.append("strategy_key = %s")
                params.append(strategy_key)
            cur.execute(
                f"""
                SELECT account_scope, market, symbol, side, quantity, entry_price, mark_price,
                       unrealized_pnl, strategy_key, risk_state_json, status, as_of, updated_at
                FROM lysara.active_positions_state
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                tuple(params + [limit]),
            )
            result["positions"] = _rows_to_dicts(cur)

            regime_where = ""
            regime_params: List[Any] = []
            if market:
                regime_where = "WHERE market = %s"
                regime_params.append(market)
            cur.execute(
                f"""
                SELECT market, regime_label, volatility_score, trend_score, confidence,
                       recommended_params_json, observed_at, source_ref, updated_at
                FROM lysara.current_regime_state
                {regime_where}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                tuple(regime_params + [max(1, min(limit, 10))]),
            )
            result["regime"] = _rows_to_dicts(cur)

            state_where: List[str] = []
            state_params: List[Any] = []
            if symbol:
                state_where.append("symbol = %s")
                state_params.append(symbol)
            if market:
                state_where.append("market = %s")
                state_params.append(market)
            where_sql = f"WHERE {' AND '.join(state_where)}" if state_where else ""
            cur.execute(
                f"""
                SELECT source_name, market, symbol, as_of, top_signals_json, catalysts_json,
                       risk_posture, feed_freshness, confidence, metadata_json, updated_at
                FROM lysara.market_working_state
                {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                tuple(state_params + [limit]),
            )
            result["market"] = _rows_to_dicts(cur)

            review_where = ["status = 'pending'"]
            review_params: List[Any] = []
            if symbol:
                review_where.append("(symbol = %s OR symbol IS NULL)")
                review_params.append(symbol)
            if market:
                review_where.append("(market = %s OR market IS NULL)")
                review_params.append(market)
            if strategy_key:
                review_where.append("(strategy_key = %s OR strategy_key IS NULL)")
                review_params.append(strategy_key)
            cur.execute(
                f"""
                SELECT id, queue_type, symbol, strategy_key, market, priority, title,
                       details_json, status, requested_at, source_ref, external_ref
                FROM lysara.review_queue
                WHERE {' AND '.join(review_where)}
                ORDER BY priority DESC, requested_at DESC
                LIMIT %s
                """,
                tuple(review_params + [limit]),
            )
            result["review_queue"] = _rows_to_dicts(cur)
        return result

    def list_review_queue(self, *, status: str = "pending", limit: int = 50) -> Dict[str, Any]:
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, queue_type, symbol, strategy_key, market, priority, title,
                       details_json, status, requested_at, resolved_at, resolution_note, source_ref
                FROM lysara.review_queue
                WHERE status = %s
                ORDER BY priority DESC, requested_at DESC
                LIMIT %s
                """,
                (status, max(1, min(int(limit or 50), 200))),
            )
            rows = _rows_to_dicts(cur)
        return {"items": rows}

    def list_open_loops(
        self,
        *,
        status: str = "open",
        symbol: Optional[str] = None,
        strategy_key: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        where = ["status = %s"]
        params: List[Any] = [status]
        if symbol:
            where.append("(symbol = %s OR symbol IS NULL)")
            params.append(symbol.strip().upper())
        if strategy_key:
            where.append("(strategy_key = %s OR strategy_key IS NULL)")
            params.append(strategy_key.strip())
        if market:
            where.append("(market = %s OR market IS NULL)")
            params.append(market.strip().lower())
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                f"""
                SELECT id, loop_type, title, description, symbol, strategy_key, market,
                       priority, trigger_conditions_json, status, owner, opened_at, due_hint,
                       closed_at, closed_reason, source_ref
                FROM lysara.open_loops
                WHERE {' AND '.join(where)}
                ORDER BY priority DESC, opened_at DESC
                LIMIT %s
                """,
                tuple(params + [max(1, min(int(limit or 50), 200))]),
            )
            rows = _rows_to_dicts(cur)
        return {"items": rows}

    def list_theses(
        self,
        *,
        symbol: Optional[str] = None,
        strategy_key: Optional[str] = None,
        status: str = "active",
        limit: int = 50,
    ) -> Dict[str, Any]:
        where = ["status = %s"]
        params: List[Any] = [status]
        if symbol:
            where.append("(symbol = %s OR symbol IS NULL)")
            params.append(symbol.strip().upper())
        if strategy_key:
            where.append("(strategy_key = %s OR strategy_key IS NULL)")
            params.append(strategy_key.strip())
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                f"""
                SELECT id, thesis_key, scope_type, symbol, sector, strategy_key, thesis_text,
                       confidence, status, supporting_refs_json, source_note_id, created_at, updated_at
                FROM lysara.market_theses
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                tuple(params + [max(1, min(int(limit or 50), 200))]),
            )
            rows = _rows_to_dicts(cur)
        return {"items": rows}

    def _staleness_snapshot(self) -> Dict[str, Any]:
        with pooled_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT source_name, last_success_at, payload_updated_at, stale_after_seconds, error
                FROM lysara.sync_state
                """
            )
            rows = _rows_to_dicts(cur)
        now = _now_utc()
        result: Dict[str, Any] = {}
        for row in rows:
            source_name = str(row.get("source_name") or "")
            last_success_raw = row.get("last_success_at")
            last_success_dt = datetime.fromisoformat(last_success_raw) if last_success_raw else None
            stale_after_seconds = row.get("stale_after_seconds")
            stale = False
            age_seconds = None
            if last_success_dt and stale_after_seconds:
                age_seconds = int((now - last_success_dt).total_seconds())
                stale = age_seconds > int(stale_after_seconds)
            result[source_name] = {
                "last_success_at": last_success_raw,
                "payload_updated_at": row.get("payload_updated_at"),
                "stale_after_seconds": stale_after_seconds,
                "age_seconds": age_seconds,
                "stale": stale,
                "error": row.get("error") or "",
            }
        return result

    def get_context_bundle(
        self,
        *,
        query: str = "",
        query_mode: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_key: Optional[str] = None,
        market: Optional[str] = None,
        sections: Optional[Sequence[Any]] = None,
        limit: int = 12,
    ) -> Dict[str, Any]:
        query_mode = self.normalize_query_mode(query_mode, query)
        extracted_symbols = self._extract_symbol_tokens(query)
        chosen_symbol = (symbol or (extracted_symbols[0] if extracted_symbols else "")).strip().upper() or None
        chosen_market = (market or "").strip().lower() or None
        chosen_strategy = (strategy_key or "").strip() or None
        limit = max(1, min(int(limit or 12), 100))
        normalized_sections = self.normalize_context_sections(sections)
        explicit_sections = bool(normalized_sections)
        wanted_sections = set(normalized_sections or self._default_sections_for_mode(query_mode))
        bundle = self._empty_context_bundle(query_mode)

        if "working_state" in wanted_sections:
            working_limit = limit if explicit_sections or query_mode == "working" else min(limit, 6)
            bundle["working_state"] = self.get_working_state(
                symbol=chosen_symbol,
                market=chosen_market,
                strategy_key=chosen_strategy,
                limit=working_limit,
            )

        if "open_loops" in wanted_sections:
            loop_limit = limit if explicit_sections or query_mode in {"working", "open_loop"} else min(limit, 6)
            bundle["open_loops"] = {
                "loops": self.list_open_loops(
                    status="open",
                    symbol=chosen_symbol,
                    strategy_key=chosen_strategy,
                    market=chosen_market,
                    limit=loop_limit,
                )["items"],
                "review_queue": self.list_review_queue(status="pending", limit=loop_limit)["items"],
            }

        if "staleness" in wanted_sections:
            bundle["staleness"] = self._staleness_snapshot()

        need_canonical = "canonical_rules" in wanted_sections
        need_recent = "recent_operations" in wanted_sections
        need_research = "research_context" in wanted_sections
        if not any([need_canonical, need_recent, need_research]):
            return bundle

        with pooled_cursor(commit=False) as cur:
            if need_canonical:
                risk_rows: List[Dict[str, Any]] = []
                constraint_rows: List[Dict[str, Any]] = []
                strategy_rows: List[Dict[str, Any]] = []
                symbol_rows: List[Dict[str, Any]] = []
                operator_rows: List[Dict[str, Any]] = []

                canonical_full = explicit_sections or query_mode == "canonical"
                canonical_mixed = (not explicit_sections) and query_mode == "mixed"
                canonical_relevant = (not explicit_sections) and query_mode in {"working", "open_loop"}

                if canonical_full or canonical_mixed:
                    cur.execute(
                        """
                        SELECT policy_key, name, scope, limits_json, approval_thresholds_json,
                               autonomy_rules_json, status, effective_at, source_ref, updated_at
                        FROM lysara.risk_policies
                        WHERE status = 'active'
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (5 if canonical_full else min(limit, 2),),
                    )
                    risk_rows = _rows_to_dicts(cur)

                    cur.execute(
                        """
                        SELECT portfolio_key, constraint_type, constraint_value_json, status, effective_at, updated_at
                        FROM lysara.portfolio_constraints
                        WHERE status = 'active'
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (limit if canonical_full else min(limit, 2),),
                    )
                    constraint_rows = _rows_to_dicts(cur)

                    cur.execute(
                        """
                        SELECT policy_key, description, tool_permissions_json, approval_rules_json,
                               mutation_rules_json, status, updated_at
                        FROM lysara.operator_policies
                        WHERE status = 'active'
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (limit if canonical_full else min(limit, 2),),
                    )
                    operator_rows = _rows_to_dicts(cur)

                if canonical_full or canonical_mixed or (canonical_relevant and (chosen_strategy or chosen_symbol)):
                    strategy_where: List[str] = []
                    strategy_params: List[Any] = []
                    if chosen_strategy:
                        strategy_where.append("strategy_key = %s")
                        strategy_params.append(chosen_strategy)
                    elif chosen_symbol:
                        strategy_where.append("(allowed_symbols_json @> %s::jsonb OR payload_json::text ILIKE %s)")
                        strategy_params.extend([_json_dumps([chosen_symbol]), f"%{chosen_symbol}%"])
                    strategy_sql = f"WHERE {' AND '.join(strategy_where)}" if strategy_where else ""
                    cur.execute(
                        f"""
                        SELECT strategy_key, name, description, market_scope, allowed_symbols_json,
                               default_params_json, status, owner, source_ref, updated_at
                        FROM lysara.strategy_profiles
                        {strategy_sql}
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        tuple(strategy_params + [limit if canonical_full else min(limit, 3)]),
                    )
                    strategy_rows = _rows_to_dicts(cur)

                    symbol_where = ""
                    symbol_params: List[Any] = []
                    if chosen_symbol:
                        symbol_where = "WHERE symbol = %s"
                        symbol_params.append(chosen_symbol)
                    elif not canonical_full and not canonical_mixed:
                        symbol_where = "WHERE FALSE"
                    cur.execute(
                        f"""
                        SELECT symbol, asset_class, market, liquidity_profile, volatility_profile,
                               restriction_status, preferred_strategies_json, notes, updated_at
                        FROM lysara.symbol_profiles
                        {symbol_where}
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        tuple(symbol_params + [limit if canonical_full else min(limit, 3)]),
                    )
                    symbol_rows = _rows_to_dicts(cur)

                bundle["canonical_rules"] = {
                    "risk_policies": risk_rows,
                    "portfolio_constraints": constraint_rows,
                    "strategies": strategy_rows,
                    "symbol_profiles": symbol_rows,
                    "operator_policies": operator_rows,
                }

            if need_recent:
                recent_full = explicit_sections or query_mode == "episodic"
                recent_limit = limit if recent_full else min(limit, 4)
                decision_where: List[str] = []
                decision_params: List[Any] = []
                if chosen_symbol:
                    decision_where.append("(symbol = %s OR symbol IS NULL)")
                    decision_params.append(chosen_symbol)
                if chosen_market:
                    decision_where.append("(market = %s OR market IS NULL)")
                    decision_params.append(chosen_market)
                if chosen_strategy:
                    decision_where.append("(strategy_key = %s OR strategy_key IS NULL)")
                    decision_params.append(chosen_strategy)
                decision_sql = f"WHERE {' AND '.join(decision_where)}" if decision_where else ""

                cur.execute(
                    f"""
                    SELECT id, trade_ref, decision_type, symbol, strategy_key, market, regime_label,
                           rationale, risk_snapshot_json, approval_state, review_item_id, decided_by,
                           decided_at, final_status, execution_payload_json, source_ref, metadata_json
                    FROM lysara.trade_decision_log
                    {decision_sql}
                    ORDER BY decided_at DESC
                    LIMIT %s
                    """,
                    tuple(decision_params + [recent_limit]),
                )
                decision_rows = _rows_to_dicts(cur)

                cur.execute(
                    f"""
                    SELECT metric_id, trade_id, source_trade_ref, market, symbol, strategy_key, strategy_name,
                           sector, regime_label, pnl, pnl_pct, win, closed_at, metadata_json
                    FROM lysara.trade_performance
                    {decision_sql}
                    ORDER BY closed_at DESC
                    LIMIT %s
                    """,
                    tuple(decision_params + [recent_limit]),
                )
                performance_rows = _rows_to_dicts(cur)

                cur.execute(
                    f"""
                    SELECT id, override_type, symbol, strategy_key, market, reason, set_by, set_at,
                           cleared_at, source_ref, payload_json
                    FROM lysara.operator_overrides
                    {decision_sql}
                    ORDER BY set_at DESC
                    LIMIT %s
                    """,
                    tuple(decision_params + [recent_limit]),
                )
                override_rows = _rows_to_dicts(cur)

                regime_where: List[str] = []
                regime_params: List[Any] = []
                if chosen_market:
                    regime_where.append("market = %s")
                    regime_params.append(chosen_market)
                regime_sql = f"WHERE {' AND '.join(regime_where)}" if regime_where else ""
                cur.execute(
                    f"""
                    SELECT market, symbol, regime_label, volatility_score, trend_score, confidence,
                           recommended_params_json, applied, source, source_ref, observed_at
                    FROM lysara.regime_history
                    {regime_sql}
                    ORDER BY observed_at DESC
                    LIMIT %s
                    """,
                    tuple(regime_params + [recent_limit]),
                )
                regime_rows = _rows_to_dicts(cur)

                bundle["recent_operations"] = {
                    "trade_decisions": decision_rows,
                    "trade_performance": performance_rows,
                    "operator_overrides": override_rows,
                    "regime_history": regime_rows,
                }

            if need_research:
                research_full = explicit_sections or query_mode == "research"
                research_limit = limit if research_full else min(limit, 4)
                decision_where: List[str] = []
                decision_params: List[Any] = []
                if chosen_symbol:
                    decision_where.append("(symbol = %s OR symbol IS NULL)")
                    decision_params.append(chosen_symbol)
                if chosen_market:
                    decision_where.append("(market = %s OR market IS NULL)")
                    decision_params.append(chosen_market)
                if chosen_strategy:
                    decision_where.append("(strategy_key = %s OR strategy_key IS NULL)")
                    decision_params.append(chosen_strategy)
                decision_sql = f"WHERE {' AND '.join(decision_where)}" if decision_where else ""

                cur.execute(
                    f"""
                    SELECT note_type, symbol, strategy_key, market, title, content, tags_json,
                           recorded_at, source_ref
                    FROM lysara.research_notes
                    {decision_sql}
                    ORDER BY recorded_at DESC
                    LIMIT %s
                    """,
                    tuple(decision_params + [research_limit]),
                )
                research_rows = _rows_to_dicts(cur)

                thesis_where: List[str] = ["status = 'active'"]
                thesis_params: List[Any] = []
                if chosen_symbol:
                    thesis_where.append("(symbol = %s OR symbol IS NULL)")
                    thesis_params.append(chosen_symbol)
                if chosen_strategy:
                    thesis_where.append("(strategy_key = %s OR strategy_key IS NULL)")
                    thesis_params.append(chosen_strategy)
                cur.execute(
                    f"""
                    SELECT thesis_key, scope_type, symbol, sector, strategy_key, thesis_text,
                           confidence, status, supporting_refs_json, source_note_id, created_at, updated_at
                    FROM lysara.market_theses
                    WHERE {' AND '.join(thesis_where)}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    tuple(thesis_params + [research_limit]),
                )
                thesis_rows = _rows_to_dicts(cur)

                bundle["research_context"] = {
                    "theses": thesis_rows,
                    "notes": research_rows,
                }

        return bundle
