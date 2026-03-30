from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RegisteredPosition:
    symbol: str
    strategy_name: str
    side: str
    entry_price: float
    quantity: float
    stop_price: float | None = None
    target_price: float | None = None
    market: str = "crypto"
    updated_at: str | None = None


class PositionRegistry:
    def __init__(self, path: str | Path = "data/position_registry.json") -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._positions: dict[str, RegisteredPosition] = {}
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
                symbol = str(item.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                payload = dict(item)
                payload["symbol"] = symbol
                self._positions[symbol] = RegisteredPosition(**payload)
        except Exception:
            self._positions = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(item) for item in self._positions.values()]
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def register(self, position: RegisteredPosition) -> dict[str, Any]:
        symbol = str(position.symbol or "").strip().upper()
        strategy_name = str(position.strategy_name or "").strip()
        side = str(position.side or "").strip().lower()
        if not symbol or not strategy_name or side not in {"buy", "sell"}:
            return {"status": "rejected", "reason": "invalid_position"}
        with self._lock:
            existing = self._positions.get(symbol)
            if existing and existing.strategy_name != strategy_name:
                if str(existing.side or "").strip().lower() != side:
                    return {
                        "status": "rejected",
                        "reason": "conflicting_position",
                        "existing": asdict(existing),
                    }
            payload = RegisteredPosition(
                symbol=symbol,
                strategy_name=strategy_name,
                side=side,
                entry_price=float(position.entry_price or 0.0),
                quantity=float(position.quantity or 0.0),
                stop_price=float(position.stop_price) if position.stop_price is not None else None,
                target_price=float(position.target_price) if position.target_price is not None else None,
                market=str(position.market or "crypto").strip().lower() or "crypto",
                updated_at=_utcnow_iso(),
            )
            self._positions[symbol] = payload
            self._save()
            return {"status": "registered", "position": asdict(payload)}

    def can_open(self, symbol: str, side: str, strategy_name: str) -> tuple[bool, list[str]]:
        clean_symbol = str(symbol or "").strip().upper()
        clean_side = str(side or "").strip().lower()
        clean_strategy = str(strategy_name or "").strip()
        with self._lock:
            existing = self._positions.get(clean_symbol)
            if existing is None:
                return True, []
            if existing.strategy_name == clean_strategy:
                return True, []
            if str(existing.side or "").strip().lower() != clean_side:
                return False, [f"conflicting_position:{existing.strategy_name}:{existing.side}"]
            return False, [f"position_owned_by:{existing.strategy_name}"]

    def release(self, symbol: str, strategy_name: str | None = None) -> dict[str, Any]:
        clean_symbol = str(symbol or "").strip().upper()
        clean_strategy = str(strategy_name or "").strip()
        with self._lock:
            existing = self._positions.get(clean_symbol)
            if existing is None:
                return {"status": "missing"}
            if clean_strategy and existing.strategy_name != clean_strategy:
                return {"status": "ignored", "reason": "strategy_mismatch", "position": asdict(existing)}
            removed = self._positions.pop(clean_symbol)
            self._save()
            return {"status": "released", "position": asdict(removed)}

    def get_all_positions(self) -> dict[str, Any]:
        with self._lock:
            items = [asdict(item) for item in self._positions.values()]
        return {"items": items, "count": len(items)}


def _utcnow_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()


_REGISTRY: PositionRegistry | None = None


def get_registry() -> PositionRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PositionRegistry()
    return _REGISTRY
