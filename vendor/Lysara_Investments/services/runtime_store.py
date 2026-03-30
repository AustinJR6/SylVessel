from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.runtime_paths import env_or_runtime_path


_DEFAULT_PAYLOAD: dict[str, Any] = {
    "runtime": {
        "autonomous_enabled": False,
        "operator_interval_seconds": 300,
    },
    "watchlists": {
        "crypto": [],
        "stocks": [],
    },
    "strategy_candidates": [],
    "symbol_profiles": {},
}


class RuntimeStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else env_or_runtime_path("RUNTIME_STORE_PATH", "runtime_store.json")
        self._lock = threading.RLock()
        self._payload: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return deepcopy(_DEFAULT_PAYLOAD)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return deepcopy(_DEFAULT_PAYLOAD)
            return _merge_payloads(deepcopy(_DEFAULT_PAYLOAD), raw)
        except Exception:
            return deepcopy(_DEFAULT_PAYLOAD)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self._payload, ensure_ascii=True, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def get_runtime(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._payload.get("runtime") or {})

    def patch_runtime(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            runtime = self._payload.setdefault("runtime", {})
            runtime.update({key: value for key, value in (payload or {}).items() if value is not None})
            self._save()
            return deepcopy(runtime)

    def get_watchlists(self, defaults: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
        with self._lock:
            watchlists = deepcopy(self._payload.get("watchlists") or {})
        base = {"crypto": [], "stocks": []}
        if defaults:
            for market in ("crypto", "stocks"):
                base[market] = [str(symbol or "").strip().upper() for symbol in defaults.get(market) or [] if str(symbol or "").strip()]
        for market in ("crypto", "stocks"):
            if watchlists.get(market):
                base[market] = [str(symbol or "").strip().upper() for symbol in watchlists.get(market) or [] if str(symbol or "").strip()]
        return base

    def set_watchlists(self, payload: dict[str, list[str]]) -> dict[str, list[str]]:
        cleaned = {
            "crypto": [str(symbol or "").strip().upper() for symbol in payload.get("crypto") or [] if str(symbol or "").strip()],
            "stocks": [str(symbol or "").strip().upper() for symbol in payload.get("stocks") or [] if str(symbol or "").strip()],
        }
        with self._lock:
            self._payload["watchlists"] = cleaned
            self._save()
        return deepcopy(cleaned)

    def list_strategy_candidates(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._payload.get("strategy_candidates") or [])

    def queue_strategy_candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol") or "").strip().upper()
        market = str(payload.get("market") or "crypto").strip().lower() or "crypto"
        strategy_key = str(payload.get("strategy_key") or "").strip() or None
        if not symbol:
            return {"status": "rejected", "error": "symbol_required"}
        item = {
            "id": f"candidate-{symbol.lower()}-{market}",
            "symbol": symbol,
            "market": market,
            "strategy_key": strategy_key,
            "status": "pending",
            "summary": str(payload.get("summary") or "").strip(),
            "analysis": deepcopy(payload.get("analysis") or {}),
            "created_at": _utcnow_iso(),
        }
        with self._lock:
            items = self._payload.setdefault("strategy_candidates", [])
            for existing in items:
                if (
                    str(existing.get("symbol") or "").strip().upper() == symbol
                    and str(existing.get("market") or "").strip().lower() == market
                    and str(existing.get("status") or "").strip().lower() == "pending"
                ):
                    return {"status": "queued", "item": deepcopy(existing)}
            items.insert(0, item)
            del items[100:]
            self._save()
        return {"status": "queued", "item": deepcopy(item)}

    def save_symbol_profile(self, symbol: str, profile: dict[str, Any]) -> dict[str, Any]:
        clean_symbol = str(symbol or "").strip().upper()
        with self._lock:
            profiles = self._payload.setdefault("symbol_profiles", {})
            profiles[clean_symbol] = deepcopy(profile or {})
            self._save()
            return deepcopy(profiles[clean_symbol])

    def get_symbol_profile(self, symbol: str) -> dict[str, Any] | None:
        clean_symbol = str(symbol or "").strip().upper()
        with self._lock:
            profile = (self._payload.get("symbol_profiles") or {}).get(clean_symbol)
            return deepcopy(profile) if isinstance(profile, dict) else None


def _merge_payloads(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in (incoming or {}).items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            base[key] = _merge_payloads(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _utcnow_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()


_STORE: RuntimeStore | None = None


def get_runtime_store() -> RuntimeStore:
    global _STORE
    if _STORE is None:
        _STORE = RuntimeStore()
    return _STORE
