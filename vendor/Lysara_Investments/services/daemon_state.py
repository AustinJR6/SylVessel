"""
services/daemon_state.py

Shared singleton state for the Lysara Tier 1 daemon.

All running strategies, the heartbeat, and the Control API read/write
through this module — no direct cross-service references needed.

Usage:
    from services.daemon_state import get_state
    state = get_state()
    state.register_risk_manager(rm, market="crypto")
    state.pause_all("manual override")
"""
from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List


class _DaemonState:
    def __init__(self):
        self._lock = threading.Lock()

        # --- control flags ---
        self.paused: bool = False
        self.pause_reason: str = ""

        # --- runtime info ---
        self.simulation_mode: bool = True
        self.active_markets: List[str] = []
        self.regime: str = "unknown"
        self.equity: Dict[str, float] = {}      # market -> latest equity value
        self.started_at: float = time.time()
        self.last_heartbeat: float = time.time()
        self.autonomous_mode: bool = False
        self.live_trading_enabled: bool = False
        self.config: Dict[str, Any] = {}
        self.feed_freshness: Dict[str, Dict[str, float]] = {}
        self.broker_health: Dict[str, Dict[str, Any]] = {}
        self.strategy_registry: Dict[str, List[Dict[str, Any]]] = {}
        self._strategy_instances: Dict[str, Dict[str, Any]] = {}
        self.symbol_controls: Dict[str, Dict[str, bool]] = {}
        self.strategy_controls: Dict[str, Dict[str, bool]] = {}
        self._apis: Dict[str, Any] = {}
        self._db_manager: Any = None
        self._recent_intents: Dict[str, float] = {}
        self._sim_portfolio: Any = None
        self.operator_interval_seconds: int = 300
        self.last_cycle_at: str | None = None
        self.last_decision_at: str | None = None
        self.blocked_reasons: List[str] = []
        self.recent_decisions: List[Dict[str, Any]] = []
        self.override_state: Dict[str, Any] = {
            "enabled": False,
            "actor": "",
            "reason": "",
            "activated_at": None,
            "expires_at": None,
            "allowed_controls": [],
            "last_cleared_at": None,
        }

        # --- risk manager registry: market -> [RiskManager, ...] ---
        self._risk_managers: Dict[str, List[Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_risk_manager(self, rm: Any, market: str) -> None:
        """Call from BotLauncher after creating each RiskManager."""
        with self._lock:
            self._risk_managers.setdefault(market, []).append(rm)
            if market not in self.active_markets:
                self.active_markets.append(market)

    def register_api(self, api: Any, market: str) -> None:
        with self._lock:
            self._apis[market] = api
            if market not in self.active_markets:
                self.active_markets.append(market)

    def register_db_manager(self, db_manager: Any) -> None:
        with self._lock:
            self._db_manager = db_manager

    def register_sim_portfolio(self, portfolio: Any) -> None:
        with self._lock:
            self._sim_portfolio = portfolio

    def set_runtime_config(self, config: Dict[str, Any]) -> None:
        with self._lock:
            self.config = dict(config or {})
            self.simulation_mode = bool(config.get("simulation_mode", True))
            self.live_trading_enabled = bool(config.get("LIVE_TRADING_ENABLED", False))
            self.autonomous_mode = bool(config.get("ENABLE_AI_TRADE_EXECUTION", False))
            self.operator_interval_seconds = int(config.get("operator_interval_seconds", self.operator_interval_seconds))

    def apply_runtime_overrides(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            if payload.get("simulation_mode") is not None:
                self.simulation_mode = bool(payload.get("simulation_mode"))
            if payload.get("autonomous_enabled") is not None:
                self.autonomous_mode = bool(payload.get("autonomous_enabled"))
            if payload.get("operator_interval_seconds") is not None:
                self.operator_interval_seconds = max(30, int(payload.get("operator_interval_seconds")))

    def register_strategy(
        self,
        market: str,
        strategy_name: str,
        symbols: List[str],
        params: Dict[str, Any],
        *,
        instance: Any = None,
    ) -> None:
        with self._lock:
            strategies = self.strategy_registry.setdefault(market, [])
            existing = next((item for item in strategies if item.get("strategy_name") == strategy_name), None)
            enabled = bool(dict(params or {}).get("enabled", True))
            if existing is None:
                existing = {
                    "strategy_name": strategy_name,
                    "symbols": list(symbols),
                    "enabled": enabled,
                    "params": dict(params),
                }
                strategies.append(existing)
            else:
                existing["symbols"] = list(symbols)
                existing["enabled"] = enabled
                existing["params"] = dict(params)
            self.strategy_controls.setdefault(market, {})
            self.strategy_controls[market][strategy_name] = enabled
            self.symbol_controls.setdefault(market, {})
            for symbol in symbols:
                self.symbol_controls[market].setdefault(symbol, True)
            if instance is not None:
                self._strategy_instances.setdefault(market, {})[strategy_name] = instance

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause_all(self, reason: str = "manual") -> None:
        """Halt all trading by flagging every registered risk manager."""
        with self._lock:
            self.paused = True
            self.pause_reason = reason
            for rms in self._risk_managers.values():
                for rm in rms:
                    rm.drawdown_triggered = True

    def resume_all(self) -> None:
        """Re-enable trading on all registered risk managers."""
        with self._lock:
            self.paused = False
            self.pause_reason = ""
            for rms in self._risk_managers.values():
                for rm in rms:
                    rm.reset_daily_risk()

    def pause_market(self, market: str) -> int:
        """Pause a specific market. Returns number of risk managers affected."""
        with self._lock:
            affected = 0
            for rm in self._risk_managers.get(market, []):
                rm.drawdown_triggered = True
                affected += 1
        return affected

    def resume_market(self, market: str) -> int:
        """Resume a specific market. Returns number of risk managers affected."""
        with self._lock:
            affected = 0
            for rm in self._risk_managers.get(market, []):
                rm.reset_daily_risk()
                affected += 1
        return affected

    def adjust_risk(
        self,
        market: str,
        risk_per_trade: float | None = None,
        max_daily_loss: float | None = None,
    ) -> int:
        """Adjust live risk parameters. Returns number of risk managers updated."""
        with self._lock:
            updated = 0
            targets = (
                [rm for rms in self._risk_managers.values() for rm in rms]
                if market == "all"
                else self._risk_managers.get(market, [])
            )
            for rm in targets:
                if risk_per_trade is not None:
                    rm.risk_per_trade = max(0.001, min(float(risk_per_trade), 0.1))
                if max_daily_loss is not None:
                    rm.max_daily_loss = float(max_daily_loss)
                updated += 1
        return updated

    def update_strategy_params(self, market: str, strategy_name: str | None, params: Dict[str, Any]) -> int:
        with self._lock:
            updated = 0
            for strat in self.strategy_registry.get(market, []):
                if strategy_name and strat["strategy_name"] != strategy_name:
                    continue
                strat["params"].update(params)
                symbols = params.get("trade_symbols")
                if isinstance(symbols, list):
                    clean_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
                    strat["symbols"] = clean_symbols
                    instance = (self._strategy_instances.get(market) or {}).get(strat["strategy_name"])
                    if instance is not None and hasattr(instance, "set_symbols"):
                        instance.set_symbols(clean_symbols)
                    self.symbol_controls.setdefault(market, {})
                    for symbol in clean_symbols:
                        self.symbol_controls[market][symbol] = True
                updated += 1
            return updated

    def set_market_enabled(self, market: str, enabled: bool) -> None:
        with self._lock:
            self.strategy_controls.setdefault(market, {})
            for key in list(self.strategy_controls.get(market, {}).keys()):
                self.strategy_controls[market][key] = enabled

    def set_strategy_enabled(self, market: str, strategy_name: str, enabled: bool) -> None:
        with self._lock:
            self.strategy_controls.setdefault(market, {})
            self.strategy_controls[market][strategy_name] = enabled
            for strat in self.strategy_registry.get(market, []):
                if strat["strategy_name"] == strategy_name:
                    strat["enabled"] = enabled

    def set_symbol_enabled(self, market: str, symbol: str, enabled: bool) -> None:
        with self._lock:
            self.symbol_controls.setdefault(market, {})
            self.symbol_controls[market][symbol] = enabled

    def is_symbol_enabled(self, market: str, symbol: str) -> bool:
        with self._lock:
            if market in self.symbol_controls and symbol in self.symbol_controls[market]:
                return self.symbol_controls[market][symbol]
            return True

    def is_strategy_enabled(self, market: str, strategy_name: str) -> bool:
        with self._lock:
            return self.strategy_controls.get(market, {}).get(strategy_name, True)

    def get_strategy_instance(self, market: str, strategy_name: str) -> Any:
        with self._lock:
            return (self._strategy_instances.get(market) or {}).get(strategy_name)

    def get_api(self, market: str) -> Any:
        with self._lock:
            return self._apis.get(market)

    def get_db_manager(self) -> Any:
        with self._lock:
            return self._db_manager

    def get_sim_portfolio(self) -> Any:
        with self._lock:
            return self._sim_portfolio

    def update_feed(self, market: str, source: str, symbol: str | None = None) -> None:
        with self._lock:
            market_state = self.feed_freshness.setdefault(market, {})
            key = source if not symbol else f"{source}:{symbol}"
            market_state[key] = time.time()

    def set_broker_health(self, market: str, connected: bool, detail: str = "") -> None:
        with self._lock:
            self.broker_health[market] = {
                "connected": connected,
                "detail": detail,
                "updated_at": time.time(),
            }

    def mark_trade_intent(self, dedupe_key: str) -> None:
        with self._lock:
            self._recent_intents[dedupe_key] = time.time()

    def seen_trade_intent(self, dedupe_key: str, cooldown_seconds: int = 60) -> bool:
        with self._lock:
            ts = self._recent_intents.get(dedupe_key)
            if ts is None:
                return False
            return (time.time() - ts) <= cooldown_seconds

    def activate_override(self, *, actor: str, reason: str, ttl_minutes: int, allowed_controls: List[str] | None = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now.timestamp() + max(1, int(ttl_minutes)) * 60
        controls = [str(item).strip() for item in (allowed_controls or []) if str(item).strip()]
        with self._lock:
            self.override_state = {
                "enabled": True,
                "actor": actor,
                "reason": reason,
                "activated_at": now.isoformat(),
                "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
                "allowed_controls": controls or list(self.config.get("override_allowed_controls", [])),
                "last_cleared_at": self.override_state.get("last_cleared_at"),
            }
            return dict(self.override_state)

    def clear_override(self, *, actor: str = "", reason: str = "") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            previous = dict(self.override_state)
            self.override_state = {
                "enabled": False,
                "actor": actor or previous.get("actor") or "",
                "reason": reason or previous.get("reason") or "",
                "activated_at": previous.get("activated_at"),
                "expires_at": None,
                "allowed_controls": list(previous.get("allowed_controls") or []),
                "last_cleared_at": now,
            }
            return dict(self.override_state)

    def get_override_status(self) -> Dict[str, Any]:
        with self._lock:
            state = dict(self.override_state)
        expires_at = state.get("expires_at")
        if state.get("enabled") and expires_at:
            try:
                expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if expiry <= now:
                    return self.clear_override(reason="expired")
                state["ttl_seconds"] = max(0, int((expiry - now).total_seconds()))
            except Exception:
                state["ttl_seconds"] = 0
        else:
            state["ttl_seconds"] = 0
        return state

    def override_active(self, control_name: str | None = None) -> bool:
        state = self.get_override_status()
        if not state.get("enabled"):
            return False
        if control_name is None:
            return True
        allowed = {str(item).strip() for item in (state.get("allowed_controls") or []) if str(item).strip()}
        return control_name in allowed

    # ------------------------------------------------------------------
    # Telemetry updates (called by strategies / heartbeat)
    # ------------------------------------------------------------------

    def update_equity(self, market: str, value: float) -> None:
        self.equity[market] = round(value, 4)

    def set_regime(self, regime: str) -> None:
        self.regime = regime

    def heartbeat(self) -> None:
        self.last_heartbeat = time.time()

    def record_strategy_decision(
        self,
        *,
        market: str,
        strategy_name: str,
        symbol: str | None,
        action: str,
        status: str,
        reasons: List[str] | None = None,
        confidence: float | None = None,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": now_iso,
            "market": market,
            "strategy_name": strategy_name,
            "symbol": symbol,
            "action": action,
            "status": status,
            "reasons": list(reasons or []),
            "confidence": confidence,
        }
        with self._lock:
            self.last_decision_at = now_iso
            self.recent_decisions.insert(0, entry)
            del self.recent_decisions[100:]
            if entry["reasons"] and status in {"blocked", "skipped", "rejected"}:
                self.blocked_reasons = list(dict.fromkeys(entry["reasons"] + self.blocked_reasons))[:20]
            elif status in {"executed", "filled"}:
                self.blocked_reasons = [reason for reason in self.blocked_reasons if reason not in set(entry["reasons"] or [])]

    def mark_cycle(self) -> None:
        self.last_cycle_at = datetime.now(timezone.utc).isoformat()

    def set_blocked_reasons(self, reasons: List[str]) -> None:
        with self._lock:
            self.blocked_reasons = [str(reason).strip() for reason in reasons if str(reason).strip()]

    def is_paused(self) -> bool:
        return self.paused

    # ------------------------------------------------------------------
    # Snapshot (used by /status endpoint)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        heartbeat_age = max(0.0, now - self.last_heartbeat)
        return {
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "simulation_mode": self.simulation_mode,
            "live_trading_enabled": self.live_trading_enabled,
            "autonomous_mode": self.autonomous_mode,
            "active_markets": list(self.active_markets),
            "regime": self.regime,
            "equity": dict(self.equity),
            "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "last_heartbeat_at": datetime.fromtimestamp(self.last_heartbeat, tz=timezone.utc).isoformat(),
            "uptime_seconds": int(now - self.started_at),
            "last_heartbeat_ago": round(heartbeat_age, 1),
            "feed_freshness": {
                market: {key: round(now - ts, 2) for key, ts in values.items()}
                for market, values in self.feed_freshness.items()
            },
            "broker_health": dict(self.broker_health),
            "strategy_registry": {
                market: list(items) for market, items in self.strategy_registry.items()
            },
            "symbol_controls": dict(self.symbol_controls),
            "strategy_controls": dict(self.strategy_controls),
            "risk_managers": {
                market: len(rms) for market, rms in self._risk_managers.items()
            },
            "override": self.get_override_status(),
            "simulation_portfolio": self._sim_portfolio.account_snapshot() if self._sim_portfolio else None,
            "operator_interval_seconds": self.operator_interval_seconds,
            "last_cycle_at": self.last_cycle_at,
            "last_decision_at": self.last_decision_at,
            "blocked_reasons": list(self.blocked_reasons),
            "recent_decisions": list(self.recent_decisions[:25]),
        }


# Module-level singleton — import `get_state()` anywhere in the daemon.
_state = _DaemonState()


def get_state() -> _DaemonState:
    return _state
