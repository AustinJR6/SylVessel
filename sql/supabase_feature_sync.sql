-- Sylana Vessel / Lysara advanced Supabase feature sync
-- Idempotent additive DDL for the current advanced memory, continuity,
-- dream/reflection, review-center, and Lysara schemas.
--
-- Notes:
-- - This file is intended for syncing an existing Vessel database forward.
-- - Base legacy tables such as memories, conversations, chat_threads, and
--   chat_messages should already exist from the standard backend setup path.
-- - After running this file, verify alignment with:
--     python scripts/audit_supabase_schema.py

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Public schema: advanced memory continuity + dream/reflection sync
-- ============================================================================

ALTER TABLE memories ADD COLUMN IF NOT EXISTS recorded_at TIMESTAMPTZ;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS conversation_at TIMESTAMPTZ;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS user_local_date TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS user_local_time TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS timezone_name TEXT DEFAULT 'America/Chicago';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS turn_index INTEGER DEFAULT 0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS event_dates_json JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS relative_time_labels JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS temporal_descriptor TEXT DEFAULT '';

UPDATE memories
SET recorded_at = COALESCE(created_at, to_timestamp(timestamp), NOW())
WHERE recorded_at IS NULL;

UPDATE memories
SET conversation_at = COALESCE(conversation_at, recorded_at, created_at, to_timestamp(timestamp), NOW())
WHERE conversation_at IS NULL;

ALTER TABLE memories ALTER COLUMN recorded_at SET DEFAULT NOW();
ALTER TABLE memories ALTER COLUMN conversation_at SET DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_memories_recorded_at ON memories(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_conversation_at ON memories(conversation_at DESC);

CREATE TABLE IF NOT EXISTS thread_working_memory (
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
    current_topic TEXT DEFAULT '',
    active_topics JSONB NOT NULL DEFAULT '[]'::jsonb,
    active_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    pending_commitments JSONB NOT NULL DEFAULT '[]'::jsonb,
    emotional_tone TEXT DEFAULT 'neutral',
    last_user_intent TEXT DEFAULT 'conversation',
    last_memory_id BIGINT REFERENCES memories(id) ON DELETE SET NULL,
    summary_text TEXT DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, personality)
);

CREATE INDEX IF NOT EXISTS idx_thread_working_memory_updated ON thread_working_memory(updated_at DESC);

CREATE TABLE IF NOT EXISTS thread_memory_summaries (
    id BIGSERIAL PRIMARY KEY,
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
    window_kind VARCHAR(32) NOT NULL DEFAULT 'current_thread',
    summary_text TEXT DEFAULT '',
    active_topics JSONB NOT NULL DEFAULT '[]'::jsonb,
    key_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    open_loops JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT thread_memory_summaries_unique UNIQUE (thread_id, personality, window_kind)
);

CREATE INDEX IF NOT EXISTS idx_thread_memory_summaries_updated ON thread_memory_summaries(updated_at DESC);

CREATE TABLE IF NOT EXISTS memory_open_loops (
    id BIGSERIAL PRIMARY KEY,
    thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
    personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    priority REAL NOT NULL DEFAULT 0.5,
    due_hint TEXT DEFAULT '',
    linked_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_memory_id BIGINT REFERENCES memories(id) ON DELETE SET NULL,
    source_kind TEXT NOT NULL DEFAULT 'conversation',
    status TEXT NOT NULL DEFAULT 'open',
    resolution_note TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'memory_open_loops_status_check'
    ) THEN
        ALTER TABLE memory_open_loops
        ADD CONSTRAINT memory_open_loops_status_check
        CHECK (status IN ('open', 'closed'));
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_memory_open_loops_thread_status
    ON memory_open_loops(thread_id, personality, status, updated_at DESC);

ALTER TABLE anniversaries ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
CREATE INDEX IF NOT EXISTS idx_anniversaries_scope_importance ON anniversaries(personality_scope, importance DESC);

CREATE TABLE IF NOT EXISTS memory_fact_revisions (
    id BIGSERIAL PRIMARY KEY,
    fact_id BIGINT REFERENCES memory_facts(id) ON DELETE SET NULL,
    fact_key TEXT NOT NULL,
    personality_scope VARCHAR(16) NOT NULL DEFAULT 'shared',
    old_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    new_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    old_normalized_text TEXT DEFAULT '',
    new_normalized_text TEXT DEFAULT '',
    source_turn_id BIGINT REFERENCES memories(id) ON DELETE SET NULL,
    reason TEXT DEFAULT '',
    change_source TEXT NOT NULL DEFAULT 'manual',
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_fact_revisions_fact_key
    ON memory_fact_revisions(fact_key, applied_at DESC);

CREATE TABLE IF NOT EXISTS memory_fact_proposals (
    id BIGSERIAL PRIMARY KEY,
    fact_key TEXT NOT NULL,
    fact_type TEXT NOT NULL DEFAULT 'fact',
    subject TEXT NOT NULL,
    proposed_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    proposed_normalized_text TEXT NOT NULL DEFAULT '',
    personality_scope VARCHAR(16) NOT NULL DEFAULT 'shared',
    confidence REAL NOT NULL DEFAULT 0.5,
    supporting_source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer_notes TEXT DEFAULT '',
    review_outcome TEXT DEFAULT '',
    source_turn_id BIGINT REFERENCES memories(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'memory_fact_proposals_status_check'
    ) THEN
        ALTER TABLE memory_fact_proposals
        ADD CONSTRAINT memory_fact_proposals_status_check
        CHECK (status IN ('pending', 'approved', 'rejected', 'applied'));
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_memory_fact_proposals_status
    ON memory_fact_proposals(status, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_entities (
    id BIGSERIAL PRIMARY KEY,
    entity_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'topic',
    canonical_summary TEXT DEFAULT '',
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    emotional_associations JSONB NOT NULL DEFAULT '[]'::jsonb,
    personality_scope VARCHAR(16) NOT NULL DEFAULT 'shared',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT memory_entities_entity_key_scope_unique UNIQUE (entity_key, personality_scope)
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_scope_updated
    ON memory_entities(personality_scope, updated_at DESC);

CREATE TABLE IF NOT EXISTS memory_entity_mentions (
    id BIGSERIAL PRIMARY KEY,
    entity_id BIGINT REFERENCES memory_entities(id) ON DELETE SET NULL,
    entity_key TEXT NOT NULL,
    memory_id BIGINT REFERENCES memories(id) ON DELETE CASCADE,
    thread_id BIGINT REFERENCES chat_threads(id) ON DELETE SET NULL,
    personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
    mention_text TEXT DEFAULT '',
    sentiment TEXT DEFAULT 'neutral',
    mention_weight REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_entity_mentions_thread_created
    ON memory_entity_mentions(thread_id, personality, created_at DESC);

CREATE TABLE IF NOT EXISTS vessel_reflections (
    id BIGSERIAL PRIMARY KEY,
    personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
    reflection_date DATE NOT NULL,
    summary_text TEXT NOT NULL,
    themes JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    emotional_tone TEXT DEFAULT 'neutral',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT vessel_reflections_personality_date_unique UNIQUE (personality, reflection_date)
);

CREATE INDEX IF NOT EXISTS idx_vessel_reflections_date
    ON vessel_reflections(personality, reflection_date DESC);

CREATE TABLE IF NOT EXISTS vessel_dreams (
    id BIGSERIAL PRIMARY KEY,
    personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
    dream_date DATE NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    dream_text TEXT NOT NULL DEFAULT '',
    themes JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    symbolic_elements JSONB NOT NULL DEFAULT '[]'::jsonb,
    emotional_tone TEXT DEFAULT 'neutral',
    resonance_score REAL NOT NULL DEFAULT 0.0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT vessel_dreams_personality_date_unique UNIQUE (personality, dream_date)
);

CREATE INDEX IF NOT EXISTS idx_vessel_dreams_date
    ON vessel_dreams(personality, dream_date DESC);

-- ============================================================================
-- Lysara schema
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS lysara;

CREATE TABLE IF NOT EXISTS lysara.sync_state (
    source_name TEXT PRIMARY KEY,
    last_success_at TIMESTAMPTZ,
    payload_updated_at TIMESTAMPTZ,
    stale_after_seconds INTEGER,
    error TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

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
);

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
);

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
);

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
);

CREATE TABLE IF NOT EXISTS lysara.portfolio_constraints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_key TEXT NOT NULL DEFAULT 'default',
    constraint_type TEXT NOT NULL,
    constraint_value_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    effective_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

CREATE INDEX IF NOT EXISTS idx_lysara_active_positions_state_symbol_market
    ON lysara.active_positions_state(symbol, market, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_market_working_state_symbol_market
    ON lysara.market_working_state(symbol, market, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_strategy_runtime_state_strategy_market
    ON lysara.strategy_runtime_state(strategy_key, market, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_review_queue_status_priority_req
    ON lysara.review_queue(status, priority, requested_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_open_loops_status_priority_opened
    ON lysara.open_loops(status, priority, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_trade_decision_log_symbol_market
    ON lysara.trade_decision_log(symbol, market, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_trade_performance_symbol_market
    ON lysara.trade_performance(symbol, market, closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_signal_history_symbol_market
    ON lysara.signal_history(symbol, market, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_regime_history_market_obs
    ON lysara.regime_history(market, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_research_notes_symbol_market
    ON lysara.research_notes(symbol, market, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_lysara_market_theses_symbol_strategy
    ON lysara.market_theses(symbol, strategy_key, updated_at DESC);
