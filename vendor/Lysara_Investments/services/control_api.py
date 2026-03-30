"""
Versioned control API for Lysara trading operations.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from services.daemon_state import get_state
from services.ops_service import OpsService

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_auth(key: str = Security(_API_KEY_HEADER)) -> str:
    secret = (
        os.getenv("CONTROL_API_SECRET", "")
        or os.getenv("LYSARA_CONTROL_SECRET", "")
        or os.getenv("LYSARA_OPS_API_KEY", "")
    ).strip()
    if not secret:
        raise HTTPException(status_code=503, detail="CONTROL_API_SECRET is not configured")
    if not key:
        raise HTTPException(status_code=403, detail="Missing API key")
    if key != secret:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


app = FastAPI(
    title="Lysara Ops API",
    description="Versioned control and monitoring surface for the Lysara trading node",
    version="2.0.0",
)
router = APIRouter(prefix="/api/v1/ops", tags=["ops"])
ops = OpsService()


class PauseRequest(BaseModel):
    reason: str = "manual"
    market: str = "all"
    actor: str = "operator"


class RiskAdjustRequest(BaseModel):
    market: str
    actor: str = "operator"
    risk_per_trade: Optional[float] = None
    max_daily_loss: Optional[float] = None


class StrategyUpdateRequest(BaseModel):
    market: str
    actor: str = "operator"
    strategy_name: Optional[str] = None
    enabled: Optional[bool] = None
    symbol_controls: Dict[str, bool] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)


class TradeIntentRequest(BaseModel):
    actor: str = "sylana"
    source: str = "lysara_tool"
    market: str
    symbol: str
    side: str
    thesis: str
    confidence: float = 0.0
    size_hint: Optional[float] = None
    time_horizon: str = "intraday"
    dedupe_nonce: Optional[str] = None


class ResearchNoteRequest(BaseModel):
    actor: str = "sylana"
    market: str
    symbol: Optional[str] = None
    summary: str
    bullish_factors: List[str] = Field(default_factory=list)
    bearish_factors: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    horizon: str = "intraday"
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    stale_after: Optional[str] = None


class DecisionJournalRequest(BaseModel):
    mode: str = "direct_ops"
    action: str
    status: str = "recorded"
    market: Optional[str] = None
    symbol: Optional[str] = None
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)
    trade_intent_id: Optional[int] = None


class IncidentActionRequest(BaseModel):
    actor: str = "operator"


class SimulationResetRequest(BaseModel):
    actor: str = "operator"
    starting_balance: float = 1000.0


class RuntimeUpdateRequest(BaseModel):
    actor: str = "operator"
    simulation_mode: Optional[bool] = None
    autonomous_enabled: Optional[bool] = None
    operator_interval_seconds: Optional[int] = None


class WatchlistUpdateRequest(BaseModel):
    stocks: List[str] = Field(default_factory=list)
    crypto: List[str] = Field(default_factory=list)


class StrategyCandidateRequest(BaseModel):
    actor: str = "operator"
    symbol: str
    market: str = "crypto"
    strategy_key: Optional[str] = None
    summary: str = ""
    analysis: Dict[str, Any] = Field(default_factory=dict)


class OverrideActivateRequest(BaseModel):
    actor: str = "operator"
    reason: str
    ttl_minutes: Optional[int] = None
    allowed_controls: List[str] = Field(default_factory=list)


class OverrideClearRequest(BaseModel):
    actor: str = "operator"
    reason: str = ""


@router.get("/health")
def health():
    state = get_state()
    state.heartbeat()
    return {
        "status": "ok",
        "paused": state.paused,
        "simulation_mode": state.simulation_mode,
        "live_trading_enabled": state.live_trading_enabled,
        "ts": time.time(),
    }


@router.get("/status")
def status(_: str = Depends(_require_auth)):
    return ops.state.snapshot()


@router.get("/runtime")
def runtime(_: str = Depends(_require_auth)):
    return ops.get_runtime()


@router.put("/runtime")
def update_runtime(body: RuntimeUpdateRequest, _: str = Depends(_require_auth)):
    return ops.update_runtime(
        actor=body.actor,
        simulation_mode=body.simulation_mode,
        autonomous_enabled=body.autonomous_enabled,
        operator_interval_seconds=body.operator_interval_seconds,
    )


@router.get("/portfolio")
async def portfolio(_: str = Depends(_require_auth)):
    return await ops.get_portfolio()


@router.get("/positions")
def positions(market: Optional[str] = None, _: str = Depends(_require_auth)):
    return {"positions": ops.get_positions(market)}


@router.get("/trades/recent")
def recent_trades(limit: int = Query(20, ge=1, le=200), market: Optional[str] = None, _: str = Depends(_require_auth)):
    return {"trades": ops.get_recent_trades(limit=limit, market=market)}


@router.get("/trades/{trade_id}")
def trade_by_id(trade_id: str, _: str = Depends(_require_auth)):
    trade = ops.get_trade_by_id(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.get("/market-snapshot")
def market_snapshot(symbols: Optional[str] = None, _: str = Depends(_require_auth)):
    requested = [s.strip().upper() for s in symbols.split(",")] if symbols else None
    return ops.get_market_snapshot(requested)


@router.post("/refresh-feeds")
async def refresh_feeds(_: str = Depends(_require_auth)):
    return await ops.refresh_feeds()


@router.post("/operator/run-now")
async def run_operator_cycle(_: str = Depends(_require_auth)):
    result = await ops.refresh_feeds()
    return {
        "status": "completed",
        "runtime": ops.get_runtime(),
        "status_payload": ops.state.snapshot(),
        "snapshot": result.get("snapshot") or {},
    }


@router.get("/sentiment")
def sentiment_radar(symbols: Optional[str] = None, _: str = Depends(_require_auth)):
    requested = [s.strip().upper() for s in symbols.split(",")] if symbols else None
    return ops.get_sentiment_radar(requested)


@router.get("/confluence")
async def confluence(symbols: Optional[str] = None, _: str = Depends(_require_auth)):
    requested = [s.strip().upper() for s in symbols.split(",")] if symbols else None
    return await ops.get_confluence(requested)


@router.get("/event-risk")
def event_risk(symbols: Optional[str] = None, _: str = Depends(_require_auth)):
    requested = [s.strip().upper() for s in symbols.split(",")] if symbols else None
    return ops.get_event_risk(requested)


@router.get("/exposure")
def exposure(market: str = "crypto", _: str = Depends(_require_auth)):
    return ops.get_exposure(market=market)


@router.get("/override/status")
def override_status(_: str = Depends(_require_auth)):
    return ops.get_override_status()


@router.post("/override")
def activate_override(body: OverrideActivateRequest, _: str = Depends(_require_auth)):
    return ops.set_override(
        actor=body.actor,
        reason=body.reason,
        ttl_minutes=body.ttl_minutes,
        allowed_controls=body.allowed_controls,
    )


@router.post("/override/clear")
def clear_override(body: OverrideClearRequest, _: str = Depends(_require_auth)):
    return ops.clear_override(actor=body.actor, reason=body.reason)


@router.get("/incidents")
def incidents(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=200),
    _: str = Depends(_require_auth),
):
    return {"incidents": ops.get_incidents(status=status, limit=limit)}


@router.post("/incidents/{incident_id}/ack")
def acknowledge_incident(incident_id: int, body: IncidentActionRequest, _: str = Depends(_require_auth)):
    return ops.acknowledge_incident(incident_id, actor=body.actor)


@router.post("/incidents/{incident_id}/resolve")
def resolve_incident(incident_id: int, body: IncidentActionRequest, _: str = Depends(_require_auth)):
    return ops.resolve_incident(incident_id, actor=body.actor)


@router.get("/research")
def research(market: Optional[str] = None, limit: int = Query(50, ge=1, le=200), _: str = Depends(_require_auth)):
    return {"research_notes": ops.get_research_notes(market=market, limit=limit)}


@router.post("/research")
def add_research(body: ResearchNoteRequest, _: str = Depends(_require_auth)):
    return ops.record_research_note(body.actor, body.model_dump())


@router.get("/journal")
def journal(limit: int = Query(100, ge=1, le=200), _: str = Depends(_require_auth)):
    return {"entries": ops.get_decision_journal(limit=limit)}


@router.post("/journal")
def add_journal_entry(body: DecisionJournalRequest, _: str = Depends(_require_auth)):
    return ops.record_decision_journal(body.model_dump())


@router.post("/pause")
def pause(body: PauseRequest, _: str = Depends(_require_auth)):
    market = None if body.market == "all" else body.market
    return ops.pause_trading(reason=body.reason, market=market, actor=body.actor)


@router.post("/resume")
def resume(body: PauseRequest, _: str = Depends(_require_auth)):
    market = None if body.market == "all" else body.market
    return ops.resume_trading(market=market, actor=body.actor)


@router.post("/risk")
def adjust_risk(body: RiskAdjustRequest, _: str = Depends(_require_auth)):
    return ops.adjust_risk(
        market=body.market,
        actor=body.actor,
        risk_per_trade=body.risk_per_trade,
        max_daily_loss=body.max_daily_loss,
    )


@router.post("/strategy")
def update_strategy(body: StrategyUpdateRequest, _: str = Depends(_require_auth)):
    return ops.update_strategy_params(
        market=body.market,
        actor=body.actor,
        strategy_name=body.strategy_name,
        enabled=body.enabled,
        symbol_controls=body.symbol_controls,
        params=body.params,
    )


@router.get("/strategies")
def strategies(_: str = Depends(_require_auth)):
    return ops.get_strategies()


@router.get("/watchlist")
def get_watchlist(_: str = Depends(_require_auth)):
    return ops.get_watchlist()


@router.post("/watchlist")
def update_watchlist(body: WatchlistUpdateRequest, _: str = Depends(_require_auth)):
    return ops.update_watchlist(body.model_dump())


@router.post("/strategy-candidates")
def queue_strategy_candidate(body: StrategyCandidateRequest, _: str = Depends(_require_auth)):
    return ops.queue_strategy_candidate(body.model_dump())


@router.post("/trade-intents")
async def submit_trade_intent(body: TradeIntentRequest, _: str = Depends(_require_auth)):
    return await ops.submit_trade_intent(body.model_dump())


@router.post("/simulation/reset")
def reset_simulation(body: SimulationResetRequest, _: str = Depends(_require_auth)):
    return ops.reset_simulation_portfolio(actor=body.actor, starting_balance=body.starting_balance)


# B1 — Price trigger endpoints
@router.post("/triggers", dependencies=[Depends(_require_auth)])
async def create_trigger(request: Request):
    """Create a price level trigger."""
    from services.price_trigger_service import get_trigger_service, PriceTrigger
    body = await request.json()
    svc = get_trigger_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Trigger service not initialized")
    trigger = PriceTrigger(
        symbol=body["symbol"],
        trigger_type=body.get("trigger_type", "above"),
        price_level=float(body["price_level"]),
        action=body.get("action", "alert"),
        strategy=body.get("strategy", "manual"),
        quantity=float(body.get("quantity", 0)),
        note=body.get("note", ""),
        expiry_utc=body.get("expiry_utc"),
    )
    trigger_id = svc.add_trigger(trigger)
    return {"trigger_id": trigger_id, "status": "created"}


@router.get("/triggers", dependencies=[Depends(_require_auth)])
async def list_triggers(symbol: Optional[str] = None):
    """List active price triggers, optionally filtered by symbol."""
    from services.price_trigger_service import get_trigger_service
    svc = get_trigger_service()
    if not svc:
        return []
    return svc.list_triggers(symbol=symbol)


@router.delete("/triggers/{trigger_id}", dependencies=[Depends(_require_auth)])
async def delete_trigger(trigger_id: str):
    """Remove a price trigger by ID."""
    from services.price_trigger_service import get_trigger_service
    svc = get_trigger_service()
    if not svc:
        raise HTTPException(status_code=503, detail="Trigger service not initialized")
    removed = svc.remove_trigger(trigger_id)
    return {"removed": removed}


# B2 — Coin onboarding endpoint
@router.post("/onboard", dependencies=[Depends(_require_auth)])
async def onboard_coin_endpoint(request: Request):
    """Trigger AI-driven coin onboarding: profile → strategy select → activate."""
    body = await request.json()
    symbol = body.get("symbol")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")
    force_strategy = body.get("force_strategy")
    from services.coin_onboarding import onboard_coin
    result = await onboard_coin(symbol=symbol, force_strategy=force_strategy)
    return result.__dict__


# B3 — Position registry endpoint
@router.get("/positions/registry", dependencies=[Depends(_require_auth)])
async def get_registry_positions():
    """Return all open positions tracked by the position registry."""
    from services.position_registry import get_registry
    return get_registry().get_all_positions()


# B4 — Lysara storage stats
@router.get("/admin/lysara/storage-stats", dependencies=[Depends(_require_auth)])
async def lysara_storage_stats():
    """Row counts for Lysara tables."""
    tables = [
        "trades",
        "equity_snapshots",
        "incidents",
        "research_notes",
        "decision_journal",
        "trade_intents",
    ]
    counts = {}
    for table in tables:
        try:
            rows = ops.db.fetch_all(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = rows[0]["n"] if rows else 0
        except Exception:
            counts[table] = None
    return {"table_row_counts": counts}


app.include_router(router)


# Backward-compatible endpoints
@app.get("/health", tags=["legacy"])
def health_legacy():
    return health()


@app.get("/status", tags=["legacy"])
def status_legacy(_: str = Depends(_require_auth)):
    return ops.state.snapshot()


@app.get("/performance", tags=["legacy"])
def performance_legacy(_: str = Depends(_require_auth)):
    return {
        "by_market": ops.db.fetch_all(
            """
            SELECT market, side, COUNT(*) AS total_trades,
                   SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) AS losses,
                   ROUND(SUM(COALESCE(profit_loss, 0)), 4) AS total_pnl,
                   ROUND(AVG(COALESCE(profit_loss, 0)), 4) AS avg_pnl
            FROM trades
            WHERE profit_loss IS NOT NULL
            GROUP BY market, side
            ORDER BY market, side
            """
        ),
        "equity_history": ops.db.fetch_all(
            "SELECT market, total_equity, timestamp FROM equity_snapshots ORDER BY timestamp DESC LIMIT 100"
        ),
        "live_equity": get_state().equity,
    }


@app.get("/trades/recent", tags=["legacy"])
def recent_trades_legacy(limit: int = 20, _: str = Depends(_require_auth)):
    return {"trades": ops.get_recent_trades(limit=limit)}


@app.post("/pause", tags=["legacy"])
def pause_legacy(body: PauseRequest, _: str = Depends(_require_auth)):
    market = None if body.market == "all" else body.market
    return ops.pause_trading(reason=body.reason, market=market, actor=body.actor)


@app.post("/resume", tags=["legacy"])
def resume_legacy(body: PauseRequest, _: str = Depends(_require_auth)):
    market = None if body.market == "all" else body.market
    return ops.resume_trading(market=market, actor=body.actor)


@app.post("/strategy/pause", tags=["legacy"])
def pause_market_legacy(body: PauseRequest, _: str = Depends(_require_auth)):
    return ops.pause_trading(reason=body.reason, market=None if body.market == "all" else body.market, actor=body.actor)


@app.post("/strategy/resume", tags=["legacy"])
def resume_market_legacy(body: PauseRequest, _: str = Depends(_require_auth)):
    return ops.resume_trading(market=None if body.market == "all" else body.market, actor=body.actor)


@app.post("/risk/adjust", tags=["legacy"])
def adjust_risk_legacy(body: RiskAdjustRequest, _: str = Depends(_require_auth)):
    return ops.adjust_risk(
        market=body.market,
        actor=body.actor,
        risk_per_trade=body.risk_per_trade,
        max_daily_loss=body.max_daily_loss,
    )
