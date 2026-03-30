import json
from pathlib import Path

SENTIMENT_PATH = Path("dashboard/data/sentiment_cache.json")


def get_sentiment_score(symbol: str) -> float:
    """Return sentiment score for a symbol from cache, or 0.0 if unavailable."""
    if not SENTIMENT_PATH.is_file():
        return 0.0
    try:
        data = json.loads(SENTIMENT_PATH.read_text())
        symbol_key = str(symbol or "").upper()
        symbol_entry = ((data.get("symbols") or {}) if isinstance(data.get("symbols"), dict) else {}).get(symbol_key)
        if isinstance(symbol_entry, dict):
            composite = symbol_entry.get("composite") or {}
            if composite.get("score") is not None:
                return float(composite.get("score") or 0.0)

        scores = []
        for source_name in ("reddit", "newsapi", "cryptopanic", "x"):
            source_payload = data.get(source_name)
            if not isinstance(source_payload, dict):
                continue
            if symbol_key in source_payload and isinstance(source_payload.get(symbol_key), dict):
                val = (source_payload.get(symbol_key) or {}).get("score")
                if val is not None:
                    scores.append(float(val))
                continue
            val = source_payload.get("score")
            if val is not None:
                scores.append(float(val))
                continue
            for nested in source_payload.values():
                if isinstance(nested, dict) and nested.get("score") is not None:
                    scores.append(float(nested.get("score")))

        return sum(scores) / len(scores) if scores else 0.0
    except Exception:
        return 0.0
