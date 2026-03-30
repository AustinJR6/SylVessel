from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data.price_cache import get_price
from utils.runtime_paths import env_or_runtime_path


@dataclass
class PriceTrigger:
    symbol: str
    trigger_type: str
    price_level: float
    action: str = "alert"
    strategy: str = "manual"
    quantity: float = 0.0
    note: str = ""
    expiry_utc: str | None = None
    trigger_id: str | None = None
    status: str = "active"
    created_at: str | None = None
    triggered_at: str | None = None
    last_price: float | None = None
    last_checked_at: str | None = None


class PriceTriggerService:
    def __init__(
        self,
        *,
        crypto_api=None,
        sim_portfolio=None,
        simulation_mode: bool = True,
        path: str | Path | None = None,
    ) -> None:
        self.crypto_api = crypto_api
        self.sim_portfolio = sim_portfolio
        self.simulation_mode = bool(simulation_mode)
        self.path = Path(path) if path else env_or_runtime_path("PRICE_TRIGGERS_PATH", "price_triggers.json")
        self._lock = threading.RLock()
        self._prices: dict[str, float] = {}
        self._triggers: dict[str, PriceTrigger] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return
            for item in raw:
                if not isinstance(item, dict):
                    continue
                trigger = PriceTrigger(**item)
                trigger_id = str(trigger.trigger_id or "").strip()
                if trigger_id:
                    self._triggers[trigger_id] = trigger
        except Exception:
            self._triggers = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(trigger) for trigger in self._triggers.values()]
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def add_trigger(self, trigger: PriceTrigger) -> str:
        trigger_id = str(trigger.trigger_id or f"trigger-{uuid.uuid4().hex[:12]}")
        normalized = PriceTrigger(
            symbol=str(trigger.symbol or "").strip().upper(),
            trigger_type=str(trigger.trigger_type or "above").strip().lower(),
            price_level=float(trigger.price_level or 0.0),
            action=str(trigger.action or "alert").strip().lower(),
            strategy=str(trigger.strategy or "manual").strip() or "manual",
            quantity=float(trigger.quantity or 0.0),
            note=str(trigger.note or "").strip(),
            expiry_utc=trigger.expiry_utc,
            trigger_id=trigger_id,
            status="active",
            created_at=_utcnow_iso(),
        )
        with self._lock:
            self._triggers[trigger_id] = normalized
            self._save()
        return trigger_id

    def list_triggers(self, symbol: str | None = None) -> list[dict[str, Any]]:
        clean_symbol = str(symbol or "").strip().upper()
        with self._lock:
            items = [asdict(trigger) for trigger in self._triggers.values()]
        if clean_symbol:
            items = [item for item in items if str(item.get("symbol") or "").strip().upper() == clean_symbol]
        return items

    def remove_trigger(self, trigger_id: str) -> bool:
        clean_trigger_id = str(trigger_id or "").strip()
        if not clean_trigger_id:
            return False
        with self._lock:
            removed = self._triggers.pop(clean_trigger_id, None)
            if removed is not None:
                self._save()
            return removed is not None

    def update_price(self, symbol: str, price: float) -> None:
        clean_symbol = str(symbol or "").strip().upper()
        if not clean_symbol or price <= 0:
            return
        with self._lock:
            self._prices[clean_symbol] = float(price)

    async def run_trigger_loop(self, interval_seconds: int = 5) -> None:
        while True:
            try:
                await self.evaluate_triggers()
            except Exception:
                pass
            await asyncio.sleep(max(1, int(interval_seconds)))

    async def evaluate_triggers(self) -> list[dict[str, Any]]:
        fired: list[dict[str, Any]] = []
        with self._lock:
            trigger_ids = list(self._triggers.keys())
        for trigger_id in trigger_ids:
            trigger = self._get_trigger(trigger_id)
            if trigger is None or trigger.status != "active":
                continue
            price = self._resolve_price(trigger.symbol)
            if price <= 0:
                continue
            should_fire = False
            if trigger.trigger_type == "below":
                should_fire = price <= trigger.price_level
            else:
                should_fire = price >= trigger.price_level
            self._update_trigger_runtime(trigger_id, last_price=price)
            if not should_fire:
                continue
            result = await self._fire_trigger(trigger, price)
            fired.append(result)
        return fired

    async def _fire_trigger(self, trigger: PriceTrigger, price: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trigger_id": trigger.trigger_id,
            "symbol": trigger.symbol,
            "action": trigger.action,
            "price": price,
            "status": "triggered",
        }
        broker_result: dict[str, Any] | None = None
        if trigger.action in {"buy", "sell"} and trigger.quantity > 0 and self.crypto_api is not None:
            broker_result = await self.crypto_api.place_order(
                symbol=trigger.symbol,
                side=trigger.action,
                qty=float(trigger.quantity),
                order_type="MARKET",
                price=price,
                confidence=1.0,
            )
            payload["broker_result"] = broker_result
        self._mark_trigger_triggered(trigger.trigger_id or "", price)
        return payload

    def _resolve_price(self, symbol: str) -> float:
        clean_symbol = str(symbol or "").strip().upper()
        with self._lock:
            cached = self._prices.get(clean_symbol)
        if cached and cached > 0:
            return float(cached)
        entry = get_price(clean_symbol)
        if isinstance(entry, dict):
            try:
                return float(entry.get("price") or 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _mark_trigger_triggered(self, trigger_id: str, price: float) -> None:
        with self._lock:
            trigger = self._triggers.get(trigger_id)
            if trigger is None:
                return
            trigger.status = "triggered"
            trigger.last_price = float(price)
            trigger.last_checked_at = _utcnow_iso()
            trigger.triggered_at = _utcnow_iso()
            self._save()

    def _update_trigger_runtime(self, trigger_id: str, *, last_price: float) -> None:
        with self._lock:
            trigger = self._triggers.get(trigger_id)
            if trigger is None:
                return
            trigger.last_price = float(last_price)
            trigger.last_checked_at = _utcnow_iso()
            self._save()

    def _get_trigger(self, trigger_id: str) -> PriceTrigger | None:
        with self._lock:
            trigger = self._triggers.get(trigger_id)
            if trigger is None:
                return None
            return PriceTrigger(**asdict(trigger))


def _utcnow_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()


_TRIGGER_SERVICE: PriceTriggerService | None = None


def init_trigger_service(**kwargs: Any) -> PriceTriggerService:
    global _TRIGGER_SERVICE
    if _TRIGGER_SERVICE is None:
        _TRIGGER_SERVICE = PriceTriggerService(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(_TRIGGER_SERVICE, key) and value is not None:
                setattr(_TRIGGER_SERVICE, key, value)
    return _TRIGGER_SERVICE


def get_trigger_service() -> PriceTriggerService | None:
    return _TRIGGER_SERVICE
