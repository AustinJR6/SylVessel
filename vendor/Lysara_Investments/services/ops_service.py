from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.config_manager import ConfigManager
from data.price_cache import get_all, get_price
from dashboard.utils.data_access import get_sentiment_data
from api.binance_public import fetch_binance_public_price
from services.confluence_engine import ConfluenceEngine
from services.daemon_state import get_state
from services.event_risk_service import EventRiskService
from services.exposure_service import ExposureService
from services.override_service import OverrideService
from services.position_sizing_service import PositionSizingService
from services.runtime_store import get_runtime_store
from services.sentiment_service import SentimentRadarService


def _utcnow() -> datetime:
    return datetime.utcnow()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


@dataclass
class PolicyDecision:
    allowed: bool
    status: str
    reasons: list[str]
    quantity: float | None = None
    price: float | None = None


class OpsService:
    def __init__(self):
        self.state = get_state()
        self.config_manager = ConfigManager()
        self.runtime_store = get_runtime_store()
        self.sentiment_radar = SentimentRadarService()
        self.confluence_engine = ConfluenceEngine(default_symbols=list(self.state.config.get("TRADE_SYMBOLS", [])))
        self.event_risk_service = EventRiskService(self.state.config)
        self.exposure_service = ExposureService(self.state)
        self.override_service = OverrideService(self.state)
        self.position_sizing_service = PositionSizingService(self.state, exposure_service=self.exposure_service)

    @property
    def db(self):
        db = self.state.get_db_manager()
        if db is None:
            raise RuntimeError("Database manager is not registered in daemon state")
        return db

    def _sync_runtime_services(self) -> None:
        try:
            db = self.db
        except RuntimeError:
            db = None
        self.override_service.db = db
        self.position_sizing_service.db = db
        self.event_risk_service.config = dict(self.state.config or {})
        self.event_risk_service.file_path = Path(self.state.config.get("event_risk_file") or self.event_risk_service.file_path)
        self.event_risk_service.lookahead_hours = max(1, int(self.state.config.get("event_risk_lookahead_hours", self.event_risk_service.lookahead_hours)))
        self.event_risk_service.warning_threshold = float(self.state.config.get("event_risk_warning_threshold", self.event_risk_service.warning_threshold))
        self.event_risk_service.block_threshold = float(self.state.config.get("event_risk_block_threshold", self.event_risk_service.block_threshold))
        self.event_risk_service.reduction_threshold = float(self.state.config.get("event_risk_reduction_threshold", self.event_risk_service.reduction_threshold))
        self.event_risk_service.reduction_factor = float(self.state.config.get("event_risk_reduction_factor", self.event_risk_service.reduction_factor))

    def _audit(self, actor: str, event_type: str, target: str, details: dict[str, Any], status: str = "applied"):
        self.db.log_audit_event(
            actor=actor,
            event_type=event_type,
            target=target,
            status=status,
            details=details,
        )

    def _incident(self, kind: str, severity: str, message: str, market: str | None = None, details: dict[str, Any] | None = None):
        incident_id = self.db.log_incident(
            kind=kind,
            severity=severity,
            message=message,
            market=market,
            details=details or {},
        )
        return incident_id

    async def get_portfolio(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "simulation_mode": self.state.simulation_mode,
            "markets": {},
            "simulation_portfolio": self.state.get_sim_portfolio().account_snapshot() if self.state.get_sim_portfolio() else None,
        }
        for market in ("crypto", "stocks", "forex"):
            api = self.state.get_api(market)
            if not api:
                continue
            account = await self._safe_fetch_account(api, market)
            positions = await self._safe_fetch_positions(api, market)
            snapshot["markets"][market] = {
                "account": account,
                "positions": positions,
            }
        return snapshot

    async def _safe_fetch_account(self, api: Any, market: str) -> dict[str, Any]:
        try:
            if hasattr(api, "fetch_account_info"):
                data = await api.fetch_account_info()
            elif hasattr(api, "get_account_info"):
                data = await api.get_account_info()
            elif hasattr(api, "get_account"):
                data = await api.get_account()
            else:
                data = {}
            self.state.set_broker_health(market, True, "account_ok")
            return data if isinstance(data, dict) else {"raw": data}
        except Exception as exc:
            logging.error("Account fetch failed for %s: %s", market, exc)
            self.state.set_broker_health(market, False, f"account_error:{exc}")
            self._incident("broker_account_error", "warning", str(exc), market=market)
            return {"error": str(exc)}

    async def _safe_fetch_positions(self, api: Any, market: str) -> list[dict[str, Any]] | dict[str, Any]:
        try:
            if market == "crypto" and hasattr(api, "get_holdings"):
                holdings = await api.get_holdings()
                return [{"symbol": asset, "quantity": qty} for asset, qty in (holdings or {}).items()]
            if hasattr(api, "get_positions"):
                positions = await api.get_positions()
                if isinstance(positions, list):
                    normalized = []
                    for item in positions:
                        if isinstance(item, dict):
                            normalized.append(item)
                        else:
                            normalized.append(
                                {
                                    "symbol": getattr(item, "symbol", ""),
                                    "qty": getattr(item, "qty", 0),
                                    "avg_entry_price": getattr(item, "avg_entry_price", 0),
                                    "current_price": getattr(item, "current_price", 0),
                                }
                            )
                    return normalized
            if hasattr(api, "fetch_holdings"):
                holdings = await api.fetch_holdings()
                if isinstance(holdings, dict):
                    return [{"symbol": asset, "quantity": qty} for asset, qty in holdings.items()]
            return []
        except Exception as exc:
            logging.error("Position fetch failed for %s: %s", market, exc)
            self.state.set_broker_health(market, False, f"positions_error:{exc}")
            self._incident("broker_positions_error", "warning", str(exc), market=market)
            return {"error": str(exc)}

    def get_positions(self, market: str | None = None) -> list[dict[str, Any]]:
        if market:
            return self.db.fetch_all(
                """
                SELECT timestamp, symbol, side, quantity, price, market, reason, profit_loss
                FROM trades WHERE market = ? ORDER BY timestamp DESC LIMIT 100
                """,
                (market,),
            )
        return self.db.fetch_all(
            """
            SELECT timestamp, symbol, side, quantity, price, market, reason, profit_loss
            FROM trades ORDER BY timestamp DESC LIMIT 200
            """
        )

    def get_recent_trades(self, limit: int = 20, market: str | None = None) -> list[dict[str, Any]]:
        if market:
            return self.db.fetch_all(
                """
                SELECT * FROM trades WHERE market = ? ORDER BY timestamp DESC LIMIT ?
                """,
                (market, min(limit, 200)),
            )
        return self.db.fetch_all(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (min(limit, 200),),
        )

    def get_trade_by_id(self, trade_id: str) -> dict[str, Any] | None:
        trade_id_text = str(trade_id or "").strip()
        if not trade_id_text:
            return None
        return self.db.fetch_one("SELECT * FROM trades WHERE id = ?", (trade_id_text,))

    def get_market_snapshot(self, symbols: list[str] | None = None) -> dict[str, Any]:
        prices = get_all()
        if symbols:
            filtered = {key: value for key, value in prices.items() if key in {s.upper() for s in symbols}}
        else:
            filtered = prices
        sentiment = get_sentiment_data()
        feed_sources = {
            symbol: {
                "provider": str((payload or {}).get("source") or "unknown"),
                "timestamp": (payload or {}).get("time"),
                "stale": bool(_parse_ts((payload or {}).get("time")) and (_utcnow() - _parse_ts((payload or {}).get("time"))).total_seconds() > 180),
            }
            for symbol, payload in filtered.items()
            if isinstance(payload, dict)
        }
        return {
            "prices": filtered,
            "feed_freshness": self.state.snapshot().get("feed_freshness", {}),
            "feed_sources": feed_sources,
            "sentiment": sentiment,
        }

    def get_sentiment_radar(self, symbols: list[str] | None = None) -> dict[str, Any]:
        payload = self.sentiment_radar.get_radar(symbols=symbols)
        payload["simulation_mode"] = self.state.simulation_mode
        payload["active_markets"] = list(self.state.active_markets)
        return payload

    async def get_confluence(self, symbols: list[str] | None = None) -> dict[str, Any]:
        payload = await self.confluence_engine.get_confluence(symbols=symbols)
        payload["simulation_mode"] = self.state.simulation_mode
        payload["active_markets"] = list(self.state.active_markets)
        return payload

    def get_event_risk(self, symbols: list[str] | None = None) -> dict[str, Any]:
        self._sync_runtime_services()
        payload = self.event_risk_service.get_event_risk(symbols=symbols)
        payload["simulation_mode"] = self.state.simulation_mode
        payload["active_markets"] = list(self.state.active_markets)
        return payload

    def get_exposure(self, market: str = "crypto") -> dict[str, Any]:
        payload = self.exposure_service.get_exposure(market=market)
        payload["simulation_mode"] = self.state.simulation_mode
        payload["active_markets"] = list(self.state.active_markets)
        return payload

    def get_override_status(self) -> dict[str, Any]:
        return {
            **self.override_service.status(),
            "simulation_mode": self.state.simulation_mode,
        }

    def get_runtime(self) -> dict[str, Any]:
        status = self.state.snapshot()
        runtime_overrides = self.runtime_store.get_runtime()
        market_snapshot = self.get_market_snapshot()
        return {
            "simulation_mode": self.state.simulation_mode,
            "autonomous_enabled": self.state.autonomous_mode,
            "paused": bool(status.get("paused")),
            "pause_reason": status.get("pause_reason"),
            "operator_interval_seconds": int(runtime_overrides.get("operator_interval_seconds", status.get("operator_interval_seconds") or 300)),
            "last_cycle_at": status.get("last_cycle_at"),
            "last_decision_at": status.get("last_decision_at"),
            "last_feed_refresh_at": runtime_overrides.get("last_feed_refresh_at"),
            "blocked_reasons": list(status.get("blocked_reasons") or []),
            "recent_decisions": list(status.get("recent_decisions") or []),
            "feed_sources": dict(market_snapshot.get("feed_sources") or {}),
            "feed_staleness": dict(market_snapshot.get("feed_freshness") or {}),
            "feed_freshness": dict(status.get("feed_freshness") or {}),
            "source": "lysara_investments",
            "updated_at": status.get("timestamp"),
        }

    def update_runtime(
        self,
        *,
        actor: str,
        simulation_mode: bool | None = None,
        autonomous_enabled: bool | None = None,
        operator_interval_seconds: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if simulation_mode is not None:
            payload["simulation_mode"] = bool(simulation_mode)
        if autonomous_enabled is not None:
            payload["autonomous_enabled"] = bool(autonomous_enabled)
        if operator_interval_seconds is not None:
            payload["operator_interval_seconds"] = max(30, int(operator_interval_seconds))
        self.state.apply_runtime_overrides(payload)
        self.runtime_store.patch_runtime(payload)
        self._audit(actor, "update_runtime", "runtime", payload)
        return self.get_runtime()

    async def refresh_feeds(self, symbols: list[str] | None = None) -> dict[str, Any]:
        self.state.mark_cycle()
        tracked = self._tracked_symbols()
        if symbols:
            normalized = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
            tracked["crypto"] = [symbol for symbol in tracked["crypto"] if symbol in normalized]
            tracked["stocks"] = [symbol for symbol in tracked["stocks"] if symbol in normalized]
        snapshot = await self._refresh_market_snapshot(tracked)
        self.runtime_store.patch_runtime({"last_feed_refresh_at": _utcnow().isoformat()})
        return {"status": "refreshed", "snapshot": snapshot, "runtime": self.get_runtime(), "status_payload": self.state.snapshot()}

    def get_strategies(self) -> dict[str, Any]:
        rows = []
        recent = list(self.state.snapshot().get("recent_decisions") or [])
        for market, items in self.state.strategy_registry.items():
            enabled_map = self.state.strategy_controls.get(market, {})
            for item in items:
                strategy_name = str(item.get("strategy_name") or "").strip()
                if not strategy_name:
                    continue
                decision = next(
                    (
                        row for row in recent
                        if str(row.get("strategy_name") or "").strip() == strategy_name
                        and str(row.get("market") or "").strip().lower() == str(market).strip().lower()
                    ),
                    None,
                )
                rows.append(
                    {
                        "strategy_key": strategy_name,
                        "strategy_name": strategy_name,
                        "market": market,
                        "enabled": bool(enabled_map.get(strategy_name, item.get("enabled", True))),
                        "symbols": list(item.get("symbols") or []),
                        "params": dict(item.get("params") or {}),
                        "symbol_count": len(item.get("symbols") or []),
                        "last_update": (item.get("params") or {}).get("updated_at") or self.state.snapshot().get("timestamp"),
                        "last_decision": decision,
                    }
                )
        return {"items": rows}

    def get_watchlist(self) -> dict[str, Any]:
        defaults = {
            "crypto": [str(symbol or "").strip().upper() for symbol in self.state.config.get("TRADE_SYMBOLS", []) if str(symbol or "").strip()],
            "stocks": [str(symbol or "").strip().upper() for symbol in (self.state.config.get("stocks_settings", {}) or {}).get("trade_symbols", []) if str(symbol or "").strip()],
        }
        return {
            "watchlists": self.runtime_store.get_watchlists(defaults),
            "strategy_candidates": self.runtime_store.list_strategy_candidates(),
            "updated_at": _utcnow().isoformat(),
        }

    def update_watchlist(self, watchlists: dict[str, Any]) -> dict[str, Any]:
        updated = self.runtime_store.set_watchlists(
            {
                "crypto": watchlists.get("crypto") or [],
                "stocks": watchlists.get("stocks") or [],
            }
        )
        self._persist_watchlists(updated)
        return {
            "watchlists": updated,
            "strategy_candidates": self.runtime_store.list_strategy_candidates(),
            "updated_at": _utcnow().isoformat(),
        }

    def queue_strategy_candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.runtime_store.queue_strategy_candidate(payload)
        self._audit(
            str(payload.get("actor") or "operator"),
            "queue_strategy_candidate",
            f"{str(payload.get('market') or 'crypto')}:{str(payload.get('symbol') or '').upper()}",
            payload,
            status=str(result.get("status") or "queued"),
        )
        return result

    def set_override(
        self,
        *,
        actor: str,
        reason: str,
        ttl_minutes: int | None = None,
        allowed_controls: list[str] | None = None,
    ) -> dict[str, Any]:
        self._sync_runtime_services()
        return self.override_service.activate(
            actor=actor,
            reason=reason,
            ttl_minutes=ttl_minutes,
            allowed_controls=allowed_controls,
        )

    def clear_override(self, *, actor: str, reason: str = "") -> dict[str, Any]:
        self._sync_runtime_services()
        return self.override_service.clear(actor=actor, reason=reason)

    def get_incidents(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if status:
            return self.db.fetch_all(
                "SELECT * FROM incidents WHERE status = ? ORDER BY timestamp DESC LIMIT ?",
                (status, min(limit, 200)),
            )
        return self.db.fetch_all(
            "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?",
            (min(limit, 200),),
        )

    def acknowledge_incident(self, incident_id: int, actor: str = "operator") -> dict[str, Any]:
        self.db.acknowledge_incident(incident_id)
        self._audit(actor, "ack_incident", f"incident:{incident_id}", {"incident_id": incident_id})
        return {"status": "acknowledged", "incident_id": incident_id}

    def resolve_incident(self, incident_id: int, actor: str = "operator") -> dict[str, Any]:
        self.db.resolve_incident(incident_id)
        self._audit(actor, "resolve_incident", f"incident:{incident_id}", {"incident_id": incident_id})
        return {"status": "resolved", "incident_id": incident_id}

    def get_research_notes(self, market: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM research_notes"
        params: tuple[Any, ...]
        if market:
            query += " WHERE market = ?"
            params = (market,)
        else:
            params = ()
        query += " ORDER BY timestamp DESC LIMIT ?"
        params = params + (min(limit, 200),)
        rows = self.db.fetch_all(query, params)
        for row in rows:
            row["bullish_factors"] = json.loads(row.pop("bullish_factors_json", "[]"))
            row["bearish_factors"] = json.loads(row.pop("bearish_factors_json", "[]"))
            row["sources"] = json.loads(row.pop("sources_json", "[]"))
        return rows

    def record_research_note(self, actor: str, payload: dict[str, Any]) -> dict[str, Any]:
        note_id = self.db.log_research_note(
            market=str(payload.get("market") or "crypto"),
            symbol=payload.get("symbol"),
            summary=str(payload.get("summary") or ""),
            bullish_factors=list(payload.get("bullish_factors") or []),
            bearish_factors=list(payload.get("bearish_factors") or []),
            confidence=float(payload.get("confidence") or 0.0),
            horizon=str(payload.get("horizon") or "intraday"),
            sources=list(payload.get("sources") or []),
            stale_after=payload.get("stale_after"),
            actor=actor,
        )
        self._audit(actor, "record_research", f"research:{note_id}", payload)
        return {"research_id": note_id, "status": "recorded"}

    def get_decision_journal(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM decision_journal ORDER BY timestamp DESC LIMIT ?",
            (min(limit, 200),),
        )
        for row in rows:
            row["details"] = json.loads(row.pop("details_json", "{}"))
        return rows

    def record_decision_journal(self, payload: dict[str, Any]) -> dict[str, Any]:
        journal_id = self.db.log_decision_journal(
            mode=str(payload.get("mode") or "direct_ops"),
            action=str(payload.get("action") or "observe"),
            status=str(payload.get("status") or "recorded"),
            summary=str(payload.get("summary") or ""),
            market=payload.get("market"),
            symbol=payload.get("symbol"),
            details=dict(payload.get("details") or {}),
            trade_intent_id=payload.get("trade_intent_id"),
        )
        return {"journal_id": journal_id, "status": "recorded"}

    def reset_simulation_portfolio(self, actor: str = "operator", starting_balance: float | None = None) -> dict[str, Any]:
        portfolio = self.state.get_sim_portfolio()
        if portfolio is None:
            return {"status": "unavailable", "reason": "simulation_portfolio_not_registered"}
        reset_balance = float(starting_balance if starting_balance is not None else self.state.config.get("starting_balance", 1000.0))
        portfolio.reset(starting_balance=reset_balance)
        for market in self.state.active_markets:
            self.state.update_equity(market, reset_balance)
        self._audit(
            actor,
            "reset_simulation_portfolio",
            "simulation_portfolio",
            {"starting_balance": reset_balance},
        )
        return {
            "status": "reset",
            "simulation_mode": self.state.simulation_mode,
            "portfolio": portfolio.account_snapshot(),
            "holdings": portfolio.holdings_snapshot(),
        }

    def pause_trading(self, reason: str, market: str | None = None, actor: str = "operator") -> dict[str, Any]:
        if market and market != "all":
            affected = self.state.pause_market(market)
            self.state.set_market_enabled(market, False)
            target = f"market:{market}"
        else:
            self.state.pause_all(reason)
            for active_market in self.state.active_markets:
                self.state.set_market_enabled(active_market, False)
            affected = sum(len(v) for v in self.state._risk_managers.values())  # noqa: SLF001
            target = "all_markets"
        self._audit(actor, "pause_trading", target, {"reason": reason, "market": market or "all"})
        return {"status": "paused", "market": market or "all", "affected": affected, "reason": reason}

    def resume_trading(self, market: str | None = None, actor: str = "operator") -> dict[str, Any]:
        if market and market != "all":
            affected = self.state.resume_market(market)
            self.state.set_market_enabled(market, True)
            target = f"market:{market}"
        else:
            self.state.resume_all()
            for active_market in self.state.active_markets:
                self.state.set_market_enabled(active_market, True)
            affected = sum(len(v) for v in self.state._risk_managers.values())  # noqa: SLF001
            target = "all_markets"
        self._audit(actor, "resume_trading", target, {"market": market or "all"})
        return {"status": "resumed", "market": market or "all", "affected": affected}

    def adjust_risk(self, market: str, actor: str, risk_per_trade: float | None = None, max_daily_loss: float | None = None):
        updated = self.state.adjust_risk(market, risk_per_trade=risk_per_trade, max_daily_loss=max_daily_loss)
        self._audit(
            actor,
            "adjust_risk",
            f"market:{market}",
            {
                "market": market,
                "risk_per_trade": risk_per_trade,
                "max_daily_loss": max_daily_loss,
                "updated": updated,
            },
        )
        return {"status": "adjusted", "market": market, "updated": updated}

    def update_strategy_params(
        self,
        *,
        market: str,
        actor: str,
        strategy_name: str | None,
        enabled: bool | None = None,
        symbol_controls: dict[str, bool] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        updated = 0
        if enabled is not None:
            if strategy_name:
                self.state.set_strategy_enabled(market, strategy_name, enabled)
            else:
                self.state.set_market_enabled(market, enabled)
            updated += 1
        if params:
            updated += self.state.update_strategy_params(market, strategy_name, params)
        if symbol_controls:
            for symbol, is_enabled in symbol_controls.items():
                self.state.set_symbol_enabled(market, symbol, bool(is_enabled))
                updated += 1
        self._persist_market_settings(market)
        self._audit(
            actor,
            "update_strategy_params",
            f"{market}:{strategy_name or 'all'}",
            {
                "market": market,
                "strategy_name": strategy_name,
                "enabled": enabled,
                "params": params or {},
                "symbol_controls": symbol_controls or {},
            },
        )
        return {
            "status": "updated",
            "market": market,
            "strategy_name": strategy_name,
            "updated": updated,
            "strategy": next(
                (
                    item
                    for item in self.get_strategies().get("items", [])
                    if str(item.get("market") or "").strip().lower() == str(market).strip().lower()
                    and (strategy_name is None or str(item.get("strategy_name") or "").strip() == str(strategy_name).strip())
                ),
                None,
            ),
        }

    def _tracked_symbols(self) -> dict[str, list[str]]:
        watchlists = self.get_watchlist().get("watchlists") or {}
        tracked = {
            "crypto": list(watchlists.get("crypto") or []),
            "stocks": list(watchlists.get("stocks") or []),
        }
        for market, items in self.state.strategy_registry.items():
            for item in items:
                if not bool(self.state.strategy_controls.get(market, {}).get(item.get("strategy_name"), item.get("enabled", True))):
                    continue
                for symbol in item.get("symbols") or []:
                    clean = str(symbol or "").strip().upper()
                    if clean and clean not in tracked.setdefault(market, []):
                        tracked[market].append(clean)
        return tracked

    async def _refresh_market_snapshot(self, tracked: dict[str, list[str]]) -> dict[str, Any]:
        from data.price_cache import update_price

        feed_sources: dict[str, Any] = {}
        for symbol in tracked.get("crypto") or []:
            data = await fetch_binance_public_price(symbol)
            price = float(data.get("price") or 0.0)
            if price > 0:
                update_price(symbol, price, str(data.get("source") or "binance_public"))
                self.state.update_feed("crypto", str(data.get("source") or "binance_public"), symbol)
                feed_sources[symbol] = {"provider": str(data.get("source") or "binance_public"), "timestamp": _utcnow().isoformat()}
        for symbol in tracked.get("stocks") or []:
            api = self.state.get_api("stocks")
            if api is None:
                continue
            data = await api.fetch_market_price(symbol)
            price = float((data or {}).get("price") or 0.0)
            if price > 0:
                update_price(symbol, price, str((data or {}).get("source") or "alpaca_poll"))
                self.state.update_feed("stocks", str((data or {}).get("source") or "alpaca_poll"), symbol)
                feed_sources[symbol] = {"provider": str((data or {}).get("source") or "alpaca_poll"), "timestamp": _utcnow().isoformat()}
        snapshot = self.get_market_snapshot()
        snapshot["feed_sources"] = feed_sources
        return snapshot

    def _persist_watchlists(self, watchlists: dict[str, list[str]]) -> None:
        crypto_settings = self.config_manager.load_market_settings("crypto")
        crypto_settings["trade_symbols"] = list(watchlists.get("crypto") or [])
        self.config_manager.save_market_settings("crypto", crypto_settings)

        stocks_settings = self.config_manager.load_market_settings("stocks")
        stocks_settings["trade_symbols"] = list(watchlists.get("stocks") or [])
        self.config_manager.save_market_settings("stocks", stocks_settings)

        self.state.config["TRADE_SYMBOLS"] = list(watchlists.get("crypto") or [])
        self.state.config.setdefault("stocks_settings", {})["trade_symbols"] = list(watchlists.get("stocks") or [])

    def _persist_market_settings(self, market: str) -> None:
        clean_market = str(market or "").strip().lower()
        if clean_market not in {"crypto", "stocks", "forex"}:
            return
        settings = self.config_manager.load_market_settings(clean_market)
        rows = []
        enabled_symbols: list[str] = []
        for item in self.state.strategy_registry.get(clean_market, []):
            strategy_name = str(item.get("strategy_name") or "").strip()
            params = dict(item.get("params") or {})
            symbols = [str(symbol or "").strip().upper() for symbol in item.get("symbols") or [] if str(symbol or "").strip()]
            enabled = bool(self.state.strategy_controls.get(clean_market, {}).get(strategy_name, item.get("enabled", True)))
            row = {key: value for key, value in params.items() if key not in {"simulation_mode", "LIVE_TRADING_ENABLED", "market", "updated_at"}}
            row["type"] = _strategy_type_for_name(strategy_name)
            row["trade_symbols"] = symbols
            row["enabled"] = enabled
            rows.append(row)
            if enabled:
                enabled_symbols.extend(symbols)
        settings["strategies"] = rows
        settings["trade_symbols"] = list(dict.fromkeys(enabled_symbols))
        self.config_manager.save_market_settings(clean_market, settings)
        if clean_market == "crypto":
            self.state.config["TRADE_SYMBOLS"] = list(settings["trade_symbols"])
        else:
            self.state.config.setdefault(f"{clean_market}_settings", {})["trade_symbols"] = list(settings["trade_symbols"])

    async def submit_trade_intent(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._sync_runtime_services()
        market = str(payload.get("market") or "").strip().lower()
        symbol = str(payload.get("symbol") or "").strip().upper()
        side = str(payload.get("side") or "").strip().lower()
        actor = str(payload.get("actor") or "system")
        source = str(payload.get("source") or "lysara_tool")
        thesis = str(payload.get("thesis") or "").strip()
        confidence = float(payload.get("confidence") or 0.0)
        size_hint = payload.get("size_hint")
        size_hint = float(size_hint) if size_hint is not None else None
        horizon = str(payload.get("time_horizon") or "intraday")
        dedupe_key = self._build_dedupe_key(market, symbol, side, thesis, source, payload.get("dedupe_nonce"))

        policy = await self._evaluate_policy(
            market=market,
            symbol=symbol,
            side=side,
            confidence=confidence,
            size_hint=size_hint,
            dedupe_key=dedupe_key,
        )
        status = policy.status
        execution_result: dict[str, Any] = {"status": status, "reasons": list(policy.reasons)}

        if policy.allowed:
            execution_result = await self._execute_order(
                market=market,
                symbol=symbol,
                side=side,
                quantity=policy.quantity or 0.0,
                price=policy.price or 0.0,
                confidence=confidence,
                thesis=thesis,
            )
            status = "executed" if not execution_result.get("error") else "rejected"
            if status == "executed":
                self.state.mark_trade_intent(dedupe_key)
        else:
            self._incident(
                "trade_intent_rejected",
                "warning",
                "; ".join(policy.reasons),
                market=market,
                details={"market": market, "symbol": symbol, "side": side},
            )

        intent_id = self.db.log_trade_intent(
            actor=actor,
            source=source,
            market=market,
            symbol=symbol,
            side=side,
            thesis=thesis,
            confidence=confidence,
            size_hint=size_hint,
            time_horizon=horizon,
            status=status,
            dedupe_key=dedupe_key,
            policy_result={
                "allowed": policy.allowed,
                "reasons": policy.reasons,
                "quantity": policy.quantity,
                "price": policy.price,
            },
            execution_result=execution_result,
        )
        self.db.log_decision_journal(
            mode="autonomous" if actor == "sylana" else "direct_ops",
            action="submit_trade_intent",
            status=status,
            summary=f"{side.upper()} {symbol} via {source}",
            market=market,
            symbol=symbol,
            details={
                "thesis": thesis,
                "confidence": confidence,
                "reasons": policy.reasons,
                "execution_result": execution_result,
            },
            trade_intent_id=intent_id,
        )
        self._audit(
            actor,
            "submit_trade_intent",
            f"{market}:{symbol}",
            {"status": status, "side": side, "confidence": confidence, "source": source},
            status=status,
        )
        self.state.record_strategy_decision(
            market=market,
            strategy_name=source or "trade_intent",
            symbol=symbol,
            action=side,
            status="executed" if status == "executed" else ("blocked" if not policy.allowed else status),
            reasons=list(policy.reasons),
            confidence=confidence,
        )
        return {
            "trade_intent_id": intent_id,
            "status": status,
            "policy": {
                "allowed": policy.allowed,
                "reasons": policy.reasons,
                "quantity": policy.quantity,
                "price": policy.price,
            },
            "execution_result": execution_result,
        }

    def _build_dedupe_key(
        self,
        market: str,
        symbol: str,
        side: str,
        thesis: str,
        source: str,
        dedupe_nonce: Any,
    ) -> str:
        raw = "|".join([market, symbol, side, source, str(dedupe_nonce or ""), thesis[:160]])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _evaluate_policy(
        self,
        *,
        market: str,
        symbol: str,
        side: str,
        confidence: float,
        size_hint: float | None,
        dedupe_key: str,
    ) -> PolicyDecision:
        reasons: list[str] = []
        normalized_market = str(market or "").strip().lower()
        normalized_symbol = str(symbol or "").strip().upper()
        normalized_side = str(side or "").strip().lower()

        if normalized_market not in {"crypto", "stocks", "forex"}:
            return PolicyDecision(False, "rejected", ["unsupported_market"])
        if normalized_side not in {"buy", "sell"}:
            return PolicyDecision(False, "rejected", ["unsupported_side"])
        if self.state.is_paused():
            reasons.append("trading_paused")
        if self.state.seen_trade_intent(dedupe_key, cooldown_seconds=int(self.state.config.get("ops_dedupe_seconds", 60))):
            reasons.append("duplicate_intent")
        if not self.state.is_symbol_enabled(normalized_market, normalized_symbol):
            reasons.append("symbol_disabled")

        allowed_symbols = self._allowed_symbols_for_market(normalized_market)
        if allowed_symbols and normalized_symbol not in allowed_symbols:
            reasons.append("symbol_not_allowlisted")

        price_entry = get_price(normalized_symbol)
        price = float(price_entry.get("price", 0.0)) if price_entry else 0.0
        price_ts = _parse_ts(price_entry.get("time") if price_entry else None)
        max_age = int(self.state.config.get("ops_max_price_age_seconds", 180))
        if not price_entry or price <= 0:
            reasons.append("missing_market_price")
        elif price_ts and (_utcnow() - price_ts) > timedelta(seconds=max_age):
            reasons.append("stale_market_price")

        rms = self.state._risk_managers.get(normalized_market, [])  # noqa: SLF001
        if rms and any(getattr(rm, "drawdown_triggered", False) for rm in rms):
            reasons.append("risk_lockout")

        if confidence < float(self.state.config.get("ops_min_confidence", 0.55)) and not self.override_service.is_active("confidence_minimum"):
            reasons.append("confidence_below_threshold")

        if normalized_market == "crypto" and normalized_side == "buy":
            event_risk = self.event_risk_service.get_symbol_risk(normalized_symbol)
            if event_risk.get("block_new_positions") and not self.override_service.is_active("event_risk_warning"):
                reasons.append("event_risk_blocked")

        current_position_qty = self._current_sim_quantity(normalized_market, normalized_symbol)
        qty = size_hint
        if normalized_side == "sell" and qty is None:
            qty = current_position_qty

        sizing: dict[str, Any] | None = None
        if rms and price > 0:
            try:
                sizing = self.position_sizing_service.compute_order_size(
                    market=normalized_market,
                    symbol=normalized_symbol,
                    side=normalized_side,
                    price=price,
                    confidence=confidence,
                    risk_manager=rms[0],
                    price_history=[],
                    desired_qty=qty,
                    current_position_qty=current_position_qty,
                )
                qty = float(sizing.get("quantity") or 0.0)
            except Exception:
                sizing = None
                qty = None

        if qty is None or qty <= 0:
            reasons.append("invalid_position_size" if normalized_side == "buy" else "no_open_position")
        else:
            max_notional = float(self.state.config.get("ops_max_notional_per_trade", 5000.0))
            if (qty * max(price, 0.0)) > max_notional:
                reasons.append("max_notional_exceeded")
            if normalized_side == "buy" and sizing:
                projected_position = float(sizing.get("projected_position_pct") or 0.0)
                projected_gross = float(sizing.get("projected_gross_exposure_pct") or 0.0)
                max_single = float((sizing.get("exposure") or {}).get("max_single_position_pct") or 0.0)
                max_gross = float((sizing.get("exposure") or {}).get("max_total_gross_exposure_pct") or 0.0)
                if max_single > 0 and projected_position > max_single + 1e-9:
                    reasons.append("single_position_cap_exceeded")
                if max_gross > 0 and projected_gross > max_gross + 1e-9:
                    reasons.append("gross_exposure_cap_exceeded")

        if reasons:
            return PolicyDecision(False, "rejected", reasons, quantity=qty, price=price)
        return PolicyDecision(True, "approved", [], quantity=qty, price=price)

    def _allowed_symbols_for_market(self, market: str) -> set[str]:
        config = self.state.config or {}
        if market == "crypto":
            return {str(s).upper() for s in config.get("TRADE_SYMBOLS", [])}
        if market == "stocks":
            return {str(s).upper() for s in config.get("stocks_settings", {}).get("trade_symbols", [])}
        if market == "forex":
            return {str(s).upper() for s in config.get("forex_settings", {}).get("trade_symbols", [])}
        return set()

    def _current_sim_quantity(self, market: str, symbol: str) -> float:
        if market != "crypto":
            return 0.0
        portfolio = self.state.get_sim_portfolio()
        if portfolio is None:
            return 0.0
        holdings = portfolio.holdings_snapshot()
        asset_code = str(symbol or "").upper().split("-", 1)[0]
        return float(holdings.get(symbol, holdings.get(asset_code, 0.0)) or 0.0)

    async def _execute_order(
        self,
        *,
        market: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        confidence: float,
        thesis: str,
    ) -> dict[str, Any]:
        api = self.state.get_api(market)
        if api is None:
            return {"error": "api_not_registered"}
        try:
            if market == "crypto":
                result = await api.place_order(
                    symbol=symbol,
                    side=side,
                    qty=quantity,
                    order_type="MARKET",
                    confidence=confidence,
                )
            elif market == "stocks":
                result = await api.place_order(
                    symbol=symbol,
                    side=side,
                    qty=quantity,
                    type="market",
                    price=price,
                    confidence=confidence,
                )
            else:
                units = quantity if side == "buy" else -quantity
                result = await api.place_order(
                    instrument=symbol,
                    units=units,
                    order_type="MARKET",
                    price=price,
                )
            self.db.log_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type="market",
                status=str(result.get("status") or result.get("filled_qty") or "submitted"),
                market=market,
            )
            broker_status = str(
                result.get("status")
                or result.get("state")
                or result.get("message")
                or ""
            ).lower()
            if result.get("error") or broker_status in {"rejected", "blocked", "canceled", "cancelled"}:
                return {
                    "error": result.get("reason") or result.get("message") or broker_status or "order_rejected",
                    "market": market,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "broker_result": result,
                }
            return {"market": market, "symbol": symbol, "side": side, "quantity": quantity, "price": price, "broker_result": result, "thesis": thesis}
        except Exception as exc:
            logging.error("Order execution failed for %s %s: %s", market, symbol, exc)
            self._incident("order_execution_error", "critical", str(exc), market=market, details={"symbol": symbol, "side": side})
            return {"error": str(exc)}


def _strategy_type_for_name(strategy_name: str) -> str:
    mapping = {
        "MomentumStrategy": "momentum",
        "MeanReversionStrategy": "mean_reversion",
        "MicroScalpingStrategy": "micro_scalping",
        "PairsTradingStrategy": "pairs_trading",
        "AIMomentumFusion": "ai_momentum_fusion",
        "CryptoScalper": "crypto_scalper",
        "SwingTradingStrategy": "swing",
        "StockMomentumStrategy": "stock_momentum",
        "ForexRSITrendStrategy": "rsi_trend",
        "BreakoutStrategy": "breakout",
        "ForexScalpingStrategy": "forex_scalping",
        "EarningsPlayStrategy": "earnings_play",
    }
    clean_name = str(strategy_name or "").strip()
    return mapping.get(clean_name, clean_name.lower())
