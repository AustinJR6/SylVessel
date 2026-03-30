from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


class EventRiskService:
    def __init__(self, config: dict | None = None, file_path: str | Path | None = None):
        self.config = config or {}
        self.file_path = Path(file_path) if file_path else Path(self.config.get("event_risk_file") or "dashboard/data/event_risk_cache.json")
        self.lookahead_hours = max(1, int(self.config.get("event_risk_lookahead_hours", 24)))
        self.warning_threshold = float(self.config.get("event_risk_warning_threshold", 0.45))
        self.block_threshold = float(self.config.get("event_risk_block_threshold", 0.7))
        self.reduction_threshold = float(self.config.get("event_risk_reduction_threshold", 0.85))
        self.reduction_factor = float(self.config.get("event_risk_reduction_factor", 0.5))

    def _load_cache(self) -> dict[str, Any]:
        if not self.file_path.is_file():
            return {}
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def _time_weight(self, starts_at: datetime) -> float:
        hours_until = max((starts_at - _utc_now()).total_seconds() / 3600.0, 0.0)
        if hours_until <= 1.0:
            return 1.0
        return max(0.2, 1.0 - ((hours_until / max(self.lookahead_hours, 1)) * 0.8))

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        title = str(event.get("title") or "").strip()
        starts_at = _parse_dt(event.get("starts_at"))
        if not title or starts_at is None:
            return None
        impact_score = max(0.0, min(float(event.get("impact_score") or 0.0), 1.0))
        return {
            "id": str(event.get("id") or f"event:{title}:{starts_at.isoformat()}"),
            "title": title,
            "category": str(event.get("category") or "event").strip(),
            "source": str(event.get("source") or "unknown").strip(),
            "starts_at": starts_at.isoformat(),
            "severity": str(event.get("severity") or "medium").strip(),
            "impact_score": impact_score,
            "scope": str(event.get("scope") or "crypto").strip(),
            "symbols": [str(symbol).upper() for symbol in (event.get("symbols") or []) if str(symbol).strip()],
            "tags": [str(tag).strip() for tag in (event.get("tags") or []) if str(tag).strip()],
            "hours_until": round(max((starts_at - _utc_now()).total_seconds() / 3600.0, 0.0), 2),
        }

    def _event_applies(self, symbol: str, event: dict[str, Any]) -> bool:
        normalized_symbol = str(symbol or "").upper()
        if event.get("scope") == "macro":
            return True
        symbols = {str(item).upper() for item in (event.get("symbols") or []) if str(item).strip()}
        if normalized_symbol in symbols:
            return True
        asset_code = normalized_symbol.split("-", 1)[0]
        return f"{asset_code}-USD" in symbols

    def _score_symbol(self, symbol: str, events: list[dict[str, Any]]) -> dict[str, Any]:
        applicable = [event for event in events if self._event_applies(symbol, event)]
        weighted = []
        for event in applicable:
            starts_at = _parse_dt(event.get("starts_at"))
            if not starts_at:
                continue
            weighted.append(min(0.95, float(event.get("impact_score") or 0.0) * self._time_weight(starts_at)))

        if weighted:
            aggregate_risk = 1.0
            for item in weighted:
                aggregate_risk *= (1.0 - item)
            risk_score = round(1.0 - aggregate_risk, 4)
        else:
            risk_score = 0.0

        if risk_score >= self.reduction_threshold:
            action = "reduce_positions"
        elif risk_score >= self.block_threshold:
            action = "avoid_new_positions"
        elif risk_score >= self.warning_threshold:
            action = "warn"
        else:
            action = "normal"

        upcoming = sorted(applicable, key=lambda item: str(item.get("starts_at") or ""))[:5]
        primary_key = upcoming[0]["id"] if upcoming else None
        reasons = [f"{event['title']} @ {event['starts_at']}" for event in upcoming[:3]]
        return {
            "symbol": symbol,
            "risk_score": risk_score,
            "action": action,
            "block_new_positions": action in {"avoid_new_positions", "reduce_positions"},
            "reduce_position_pct": self.reduction_factor if action == "reduce_positions" else 0.0,
            "warning_threshold": self.warning_threshold,
            "block_threshold": self.block_threshold,
            "reduction_threshold": self.reduction_threshold,
            "primary_event_key": primary_key,
            "upcoming_events": upcoming,
            "reasons": reasons,
        }

    def build_snapshot(self, *, provider_events: list[dict[str, Any]], symbols: list[str], configured_providers: list[str]) -> dict[str, Any]:
        normalized_events = [item for item in (self._normalize_event(event) for event in provider_events) if item]
        now = _utc_now()
        window_end = now.timestamp() + (self.lookahead_hours * 3600)
        normalized_events = [
            item
            for item in normalized_events
            if (parsed := _parse_dt(item.get("starts_at"))) is not None and parsed.timestamp() <= window_end
        ]
        normalized_events.sort(key=lambda item: str(item.get("starts_at") or ""))

        symbol_rows = [
            self._score_symbol(str(symbol).upper(), normalized_events)
            for symbol in [str(symbol).upper() for symbol in symbols if str(symbol).strip()]
        ]
        symbol_rows.sort(key=lambda item: float(item.get("risk_score") or 0.0), reverse=True)

        return {
            "updated_at": now.isoformat(),
            "configured_providers": [str(provider) for provider in configured_providers if str(provider).strip()],
            "lookahead_hours": self.lookahead_hours,
            "warning_threshold": self.warning_threshold,
            "block_threshold": self.block_threshold,
            "reduction_threshold": self.reduction_threshold,
            "reduction_factor": self.reduction_factor,
            "events": normalized_events,
            "symbols": symbol_rows,
        }

    def get_event_risk(self, symbols: list[str] | None = None) -> dict[str, Any]:
        data = self._load_cache()
        requested = [str(symbol).upper() for symbol in (symbols or []) if str(symbol).strip()]
        rows = data.get("symbols") if isinstance(data.get("symbols"), list) else []
        events = data.get("events") if isinstance(data.get("events"), list) else []

        if requested:
            rows = [row for row in rows if str((row or {}).get("symbol") or "").upper() in requested]

        rows = sorted(rows, key=lambda item: float((item or {}).get("risk_score") or 0.0), reverse=True)
        return {
            "updated_at": data.get("updated_at"),
            "configured_providers": data.get("configured_providers") or [],
            "lookahead_hours": int(data.get("lookahead_hours") or self.lookahead_hours),
            "warning_threshold": float(data.get("warning_threshold") or self.warning_threshold),
            "block_threshold": float(data.get("block_threshold") or self.block_threshold),
            "reduction_threshold": float(data.get("reduction_threshold") or self.reduction_threshold),
            "reduction_factor": float(data.get("reduction_factor") or self.reduction_factor),
            "events": events[:20],
            "symbols": rows,
        }

    def get_symbol_risk(self, symbol: str) -> dict[str, Any]:
        normalized = str(symbol or "").strip().upper()
        payload = self.get_event_risk([normalized])
        rows = payload.get("symbols") or []
        if rows:
            return rows[0]
        return {
            "symbol": normalized,
            "risk_score": 0.0,
            "action": "normal",
            "block_new_positions": False,
            "reduce_position_pct": 0.0,
            "warning_threshold": self.warning_threshold,
            "block_threshold": self.block_threshold,
            "reduction_threshold": self.reduction_threshold,
            "primary_event_key": None,
            "upcoming_events": [],
            "reasons": [],
        }
