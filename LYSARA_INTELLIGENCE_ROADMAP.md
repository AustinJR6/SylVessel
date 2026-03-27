# Lysara Intelligence Roadmap

## Purpose
This file is the working roadmap for evolving Lysara from a supervised simulation bot into a richer operator system that Sylana can run with better signals, better sizing, better visibility, and a controlled override model.

It is intended to be the implementation reference during the next development phases across:
- `Sylana_Vessel`
- `Lysara_Investments`
- `Vessel_App/sylana-vessel-app`

## Locked Scope
- Crypto only for the first implementation wave
- Simulation first
- App UI included in the first wave
- Arbitrage deferred until the rest of the stack is stable
- Override mode allowed, but hard circuit breakers remain non-bypassable
- Discord and Telegram ingestion must be authorized-channel only

## Hard Rules
These controls remain non-bypassable even during operator override mode:
- Daily loss limit
- Max single-position concentration
- Max total portfolio heat / concentration
- Stale or missing market data
- Broken broker or account connectivity
- Insufficient funds or insufficient holdings
- Malformed price, malformed quantity, or impossible order sizing
- Simulation/live mode mismatch

These controls can be relaxed during a user-enabled override window:
- Trade cooldown
- Confidence minimum
- Confluence minimum
- Approval requirement
- Event-risk warning threshold
- Sentiment threshold

## Default Override Policy
- Override is user-enabled only
- Override requires a reason
- Override is time-limited
- Default TTL: 15 minutes
- Override state must be visible in the app
- Every override activation, use, and expiration must be journaled

## Phase 1: Provider Plumbing And Signal Ingestion
### Goal
Create a first-class crypto intelligence layer that Sylana can read from directly.

### Lysara_Investments Tasks
- Add `services/sentiment_service.py`
- Add source adapters for:
  - NewsAPI
  - CryptoPanic
  - Reddit
  - X/Twitter
  - Whale-alert provider
- Add cached symbol sentiment snapshots with:
  - sentiment score
  - mention velocity
  - source coverage
  - anomaly flags
  - confidence score
- Add background refresh scheduling in `services/background_tasks.py`
- Extend `services/ops_service.py` with `get_sentiment_radar()`
- Expose `GET /api/v1/ops/sentiment` in `services/control_api.py`
- Extend config parsing in `config/config_manager.py`
- Extend `.env.example` with the new provider env vars

### Sylana_Vessel Tasks
- Extend `core/lysara_ops.py` with `get_sentiment()`
- Add `/api/lysara/sentiment` proxy in `server.py`
- Inject sentiment radar summaries into Lysara prompt context when the `lysara` tool is active
- Expand `LYSARA.md` with signal interpretation rules

### Vessel_App Tasks
- Add a sentiment radar panel to `app/(tabs)/lysara.tsx`
- Add service client functions in `services/lysara.ts`
- Add types in `types/index.ts`
- Show:
  - symbol
  - sentiment score
  - velocity
  - source confidence
  - freshness

## Phase 2: Multi-Timeframe Confluence
### Goal
Give Sylana a structured view of crypto market alignment across timeframes instead of relying on single-point snapshots.

### Lysara_Investments Tasks
- Add public candle fetch support, likely in `api/binance_public.py` or a new `api/binance_candles.py`
- Add `services/confluence_engine.py`
- Compute:
  - 1m / 5m / 15m / 1h / 4h trend alignment
  - support and resistance
  - breakout probability
  - mean-reversion probability
  - normalized confluence score
- Expose `GET /api/v1/ops/confluence`

### Sylana_Vessel Tasks
- Extend `core/lysara_ops.py` with `get_confluence()`
- Add `/api/lysara/confluence` proxy in `server.py`
- Inject confluence summaries into Lysara runtime context

### Vessel_App Tasks
- Add a confluence panel to the Lysara tab
- Show per-symbol timeframe alignment, key levels, and composite score

## Phase 3: Advanced Sizing, Exposure, And Override
### Goal
Move from basic risk sizing to layered, explainable sizing and portfolio heat control.

### Lysara_Investments Tasks
- Add `services/position_sizing_service.py`
- Add `services/exposure_service.py`
- Add `services/override_service.py`
- Extend `services/daemon_state.py` to track override state
- Upgrade the crypto strategy in `strategies/crypto/momentum.py` to use:
  - capped Kelly fraction
  - volatility scaling
  - confidence scaling
  - drawdown scaling
  - correlation-aware portfolio heat
  - override-aware gating
- Expose:
  - `GET /api/v1/ops/exposure`
  - `GET /api/v1/ops/override/status`
  - `POST /api/v1/ops/override`
  - `POST /api/v1/ops/override/clear`

### Sylana_Vessel Tasks
- Extend `core/lysara_ops.py` with exposure and override methods
- Add matching proxies in `server.py`
- Update `RISK.md` with hard-vs-soft control definitions
- Update `LYSARA.md` with override operating rules

### Vessel_App Tasks
- Add override controls to the Lysara tab
- Show:
  - override state
  - override TTL countdown
  - override actor
  - override reason
  - portfolio heat
  - concentration by symbol

## Phase 4: Event Risk Modeling
### Goal
Make Sylana event-aware so the system can reduce aggression around macro and crypto-specific catalysts.

### Lysara_Investments Tasks
- Add `services/event_risk_service.py`
- Integrate:
  - economic calendar feed
  - crypto event feed
  - volatility-risk labeling
- Expose `GET /api/v1/ops/event-risk`
- Apply pre-event risk reductions and no-new-position rules in the crypto strategy

### Sylana_Vessel Tasks
- Extend `core/lysara_ops.py` with `get_event_risk()`
- Add `/api/lysara/event-risk` proxy in `server.py`
- Add short event-risk summaries to the Lysara runtime context

### Vessel_App Tasks
- Add an event banner and event-risk timeline to the Lysara tab
- Highlight:
  - upcoming high-risk windows
  - active suppression windows
  - rationale for blocked trades

## Phase 5: Validation And Simulation Gate
### Goal
Prove the system behaves correctly before any expansion of authority.

### Lysara_Investments Tasks
- Add tests for:
  - sentiment aggregation
  - confluence scoring
  - override TTL
  - hard breaker enforcement
  - sizing caps
  - event-risk gating

### Sylana_Vessel Tasks
- Extend `scripts/validate_lysara_safety.py` for:
  - override behavior
  - sentiment endpoint health
  - confluence endpoint health
  - event-risk endpoint health
  - hard breaker non-bypassability

### Release Gate
- Multi-day simulation with no critical safety failures
- Clean override audit trail
- No stale-data execution
- No impossible sizing
- No silent failures in app or backend

## Phase 6: Arbitrage (Deferred)
### Goal
Add a separate arbitrage subsystem after the main trading stack is stable.

### Lysara_Investments Tasks
- Add `services/arbitrage_service.py`
- Compare spreads across selected venues
- Include:
  - fee-adjusted spread
  - slippage-adjusted spread
  - gas-adjusted profitability
  - depth checks
  - stale quote rejection
- Keep this isolated from the momentum strategy

### Required Constraint
- Arbitrage starts in simulation only

## Proposed Credential Inventory
These are the credentials and account inputs needed if we implement the full roadmap.

### A. Existing Core Runtime Credentials
#### Sylana_Vessel server `.env`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `SUPABASE_DB_URL`
- `MEMORY_ENCRYPTION_KEY`
- `BRAVE_SEARCH_API_KEY`
- `LYSARA_OPS_BASE_URL`
- `LYSARA_OPS_API_KEY`
- `LYSARA_SIMULATION_MODE`

#### Lysara_Investments `.env`
- `CONTROL_API_SECRET`
- `OPENAI_API_KEY`
- `ROBINHOOD_API_KEY`
- `ROBINHOOD_PRIVATE_KEY`
- `ROBINHOOD_PUBLIC_KEY`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`
- `SIMULATION_MODE`
- `ENABLE_AI_TRADE_EXECUTION`
- `LIVE_TRADING_ENABLED`
- `STARTING_BALANCE`

### B. Sentiment And Attention Feeds
#### News
- `NEWSAPI_KEY`
- `CRYPTOPANIC_KEY`

#### Reddit
- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`
- `REDDIT_SUBREDDITS`

#### X / Twitter
If we use read APIs only, plan to support:
- `X_BEARER_TOKEN`

If the chosen endpoint requires full app auth, plan to support:
- `X_API_KEY`
- `X_API_KEY_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `X_CLIENT_ID`
- `X_CLIENT_SECRET`

#### Whale / Large Transfer Tracking
Recommended first integration:
- `WHALE_ALERT_API_KEY`

Optional later alternatives:
- `GLASSNODE_API_KEY`
- `NANSEN_API_KEY`

### C. Authorized Community Feeds
Only for channels you own or explicitly authorize.

#### Discord
- `DISCORD_BOT_TOKEN`
- `DISCORD_ALLOWED_GUILD_IDS`
- `DISCORD_ALLOWED_CHANNEL_IDS`

#### Telegram
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_CHAT_IDS`
- `TELEGRAM_ALLOWED_CHANNEL_IDS`

### D. Event And Calendar Feeds
For crypto-only event risk, plan to support one macro feed and one crypto-event feed.

#### Macro / Economic Calendar
Planned env names:
- `TRADINGECONOMICS_API_KEY`
- `TRADINGECONOMICS_API_SECRET`

Alternative if another provider is chosen:
- `FINNHUB_API_KEY`

#### Crypto Event Calendar
Planned env names:
- `COINMARKETCAL_API_KEY`

### E. Arbitrage Expansion Credentials
Public monitoring can start without keys for some venues, but execution and authenticated depth data will require exchange credentials.

#### Robinhood
- `ROBINHOOD_API_KEY`
- `ROBINHOOD_PRIVATE_KEY`

#### Coinbase Advanced Trade
- `COINBASE_API_KEY`
- `COINBASE_SECRET_KEY`

#### Binance / Binance.US
- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`

#### Optional Additional CEX
If added later, expect venue-specific API key and secret pairs.

#### On-Chain / DEX
If we eventually price or simulate DEX routes:
- `RPC_URL`
- `NETWORK_ID`
- `CHAIN_ID`
- `CDP_API_KEY_ID`
- `CDP_API_KEY_SECRET`
- `CDP_WALLET_SECRET`

## Where Secrets Belong
- `Vessel_App`: no broker or provider secrets
- `Sylana_Vessel` droplet `.env`: AI, memory, search, and Lysara control credentials
- `Lysara_Investments` droplet `.env`: broker, sentiment, event, and control API credentials
- EAS: only public app configuration such as the backend base URL

## User Actions Required
- Create or obtain the provider accounts listed above
- Decide which X/Twitter auth mode is available to you
- Decide which macro calendar provider you want to use
- Decide whether you want Whale Alert, Glassnode, or Nansen for large-transfer tracking
- Provide authorized Discord guild/channel IDs if Discord ingestion is enabled
- Provide authorized Telegram chat/channel IDs if Telegram ingestion is enabled
- Confirm the override TTL policy if it changes from the default 15 minutes

## Recommended Implementation Order
1. Phase 1: sentiment and source plumbing
2. Phase 2: confluence
3. Phase 3: advanced sizing, exposure, override
4. Phase 4: event risk
5. Phase 5: validation
6. Phase 6: arbitrage

## Current Status
- Existing simulation, guardrails, reconciliation, and app-side Lysara controls are already in place
- Binance public market data is already feeding crypto simulation prices
- Crypto sizing, cooldown, stop-loss, and take-profit hardening has already been implemented
- This roadmap covers the next expansion wave
