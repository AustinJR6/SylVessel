from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.sentiment import symbol_display_name


class SentimentRadarService:
    def __init__(self, file_path: str | Path | None = None):
        self.file_path = Path(file_path) if file_path else Path("dashboard/data/sentiment_cache.json")

    def _load_cache(self) -> dict[str, Any]:
        if not self.file_path.is_file():
            return {}
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _normalize_symbol_entry(self, symbol: str, entry: dict[str, Any], configured_sources: list[str]) -> dict[str, Any]:
        composite = entry.get("composite") if isinstance(entry.get("composite"), dict) else {}
        sources = entry.get("sources") if isinstance(entry.get("sources"), dict) else {}

        if not composite and sources:
            active = [value for value in sources.values() if isinstance(value, dict)]
            total_mentions = sum(int(value.get("count") or 0) for value in active)
            raw_scores = [float(value.get("score") or 0.0) for value in active]
            composite = {
                "score": round(sum(raw_scores) / len(raw_scores), 3) if raw_scores else 0.0,
                "mention_velocity": total_mentions,
                "source_count": len(active),
                "source_coverage": round(len(active) / max(len(configured_sources), 1), 3) if configured_sources else 0.0,
                "confidence": round(min(0.95, 0.25 + min(total_mentions, 50) / 100.0), 3),
                "anomaly_flags": [],
            }

        return {
            "symbol": symbol,
            "asset": str(entry.get("asset") or symbol.split("-", 1)[0]).upper(),
            "display_name": str(entry.get("display_name") or symbol_display_name(symbol)),
            "score": float(composite.get("score") or 0.0),
            "confidence": float(composite.get("confidence") or 0.0),
            "mention_velocity": int(composite.get("mention_velocity") or 0),
            "source_count": int(composite.get("source_count") or len(sources)),
            "source_coverage": float(composite.get("source_coverage") or 0.0),
            "anomaly_flags": [str(flag) for flag in (composite.get("anomaly_flags") or []) if str(flag).strip()],
            "updated_at": entry.get("updated_at") or next((v.get("timestamp") for v in sources.values() if isinstance(v, dict) and v.get("timestamp")), None),
            "sources": sources,
        }

    def get_radar(self, symbols: list[str] | None = None) -> dict[str, Any]:
        data = self._load_cache()
        configured_sources = [str(source) for source in (data.get("configured_sources") or []) if str(source).strip()]
        raw_symbols = data.get("symbols") if isinstance(data.get("symbols"), dict) else {}

        requested = [str(symbol).upper() for symbol in (symbols or []) if str(symbol).strip()]
        items: list[dict[str, Any]] = []
        for symbol, entry in raw_symbols.items():
            normalized_symbol = str(symbol).upper()
            if requested and normalized_symbol not in requested:
                continue
            if not isinstance(entry, dict):
                continue
            items.append(self._normalize_symbol_entry(normalized_symbol, entry, configured_sources))

        if requested:
            items.sort(key=lambda item: requested.index(item["symbol"]) if item["symbol"] in requested else len(requested))
        else:
            items.sort(key=lambda item: (abs(float(item["score"])), float(item["confidence"])), reverse=True)

        updated_at = data.get("last_updated")
        if not updated_at:
            updated_at = next((item.get("updated_at") for item in items if item.get("updated_at")), None)

        return {
            "updated_at": updated_at,
            "configured_sources": configured_sources,
            "symbols": items,
        }
