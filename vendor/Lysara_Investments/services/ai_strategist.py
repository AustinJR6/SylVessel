import os
import json
import logging
import re
import hashlib
from pathlib import Path
from datetime import datetime
import asyncio
import time
from typing import Any

import openai
from dotenv import load_dotenv

# Ensure .env variables are loaded before accessing the API key.  This allows
# modules that import ai_strategist before the main configuration loads the
# environment to still pick up the key.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are a trading strategist assistant. Analyze the following context and "
    "return a clear trade recommendation in JSON format."
)

RETURN_INSTRUCTION = "Return JSON: { action, confidence (0-1), reason }"
BATCH_RETURN_INSTRUCTION = (
    "Return JSON: { decisions: ["
    "{ symbol, action, confidence (0-1), reason }"
    "] }"
)
_CONTEXT_IGNORE_KEYS = {"timestamp", "updated_at", "last_updated"}
_DECISION_CACHE: dict[str, dict[str, Any]] = {}


def _refresh_openai_api_key() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")


def _decision_model() -> str:
    return (os.getenv("OPENAI_DECISION_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"


def _max_retries() -> int:
    return max(1, int(os.getenv("AI_OPENAI_RETRIES", "2")))


def _default_min_interval() -> int:
    return max(30, int(os.getenv("AI_DECISION_MIN_INTERVAL_SECONDS", "300")))


def _default_cache_ttl() -> int:
    return max(_default_min_interval(), int(os.getenv("AI_DECISION_CACHE_TTL_SECONDS", "1800")))


def _normalize_for_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_cache(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in _CONTEXT_IGNORE_KEYS
        }
    if isinstance(value, list):
        return [_normalize_for_cache(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _context_hash(context: dict) -> str:
    normalized = _normalize_for_cache(context)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key_for_context(context: dict, cache_key: str | None = None) -> str:
    if cache_key:
        return cache_key
    symbol = str(context.get("symbol") or "").strip().upper()
    market = str(context.get("market") or "").strip().lower()
    if symbol:
        return f"{market or 'market'}:{symbol}"
    return hashlib.sha256(json.dumps(_normalize_for_cache(context), sort_keys=True).encode("utf-8")).hexdigest()


def _cache_hit(cache_key: str, fingerprint: str, min_interval_seconds: int, max_cache_age_seconds: int) -> dict | None:
    entry = _DECISION_CACHE.get(cache_key)
    if not entry:
        return None
    age = time.time() - float(entry.get("ts") or 0.0)
    if age <= min_interval_seconds:
        return {**dict(entry.get("decision") or {}), "_cached": True, "_cache_age_seconds": round(age, 1)}
    if age <= max_cache_age_seconds and entry.get("fingerprint") == fingerprint:
        return {**dict(entry.get("decision") or {}), "_cached": True, "_cache_age_seconds": round(age, 1)}
    return None


def _cache_store(cache_key: str, fingerprint: str, decision: dict) -> None:
    _DECISION_CACHE[cache_key] = {
        "fingerprint": fingerprint,
        "decision": dict(decision or {}),
        "ts": time.time(),
    }


def _hold_decision(reason: str) -> dict:
    return {"action": "hold", "confidence": 0.5, "reason": reason}

async def _call_openai(messages: list[dict]) -> str:
    """Call OpenAI ChatCompletion API and return the message content."""
    try:
        resp = await asyncio.to_thread(
            openai.chat.completions.create,
            model=_decision_model(),
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content.strip()
    except AttributeError:
        # Fallback for older openai<1.0
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=_decision_model(),
            messages=messages,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()


def _extract_json(text: str) -> dict:
    """Return the first JSON object found in a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def _log_decision(context: dict, decision: dict) -> None:
    """Append AI decisions to log file."""
    try:
        Path("logs").mkdir(exist_ok=True)
        log_path = Path("logs/ai_decisions.log")
        line = f"{datetime.utcnow().isoformat()} context={json.dumps(context)} "
        line += f"decision={json.dumps(decision)}\n"
        with open(log_path, "a") as f:
            f.write(line)
    except Exception as e:
        logging.error(f"Failed to log AI decision: {e}")


async def get_ai_trade_decision(context: dict) -> dict:
    """Analyze market context and return a trade decision."""
    return await get_ai_trade_decision_cached(context)


async def get_ai_trade_decision_cached(
    context: dict,
    *,
    cache_key: str | None = None,
    min_interval_seconds: int | None = None,
    max_cache_age_seconds: int | None = None,
) -> dict:
    enabled = os.getenv("ENABLE_AI_STRATEGY", "true").lower() in ("true", "1", "yes")
    if not enabled:
        return _hold_decision("AI disabled")

    _refresh_openai_api_key()
    if not openai.api_key:
        logging.error("OPENAI_API_KEY not set")
        return _hold_decision("No API key")

    min_interval = max(0, int(min_interval_seconds if min_interval_seconds is not None else _default_min_interval()))
    cache_ttl = max(min_interval, int(max_cache_age_seconds if max_cache_age_seconds is not None else _default_cache_ttl()))
    resolved_cache_key = _cache_key_for_context(context, cache_key)
    fingerprint = _context_hash(context)
    cached = _cache_hit(resolved_cache_key, fingerprint, min_interval, cache_ttl)
    if cached:
        return cached

    user_content = "\n".join(f"{k}: {v}" for k, v in context.items())
    user_content += f"\n\n{RETURN_INSTRUCTION}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(_max_retries()):
        try:
            text = await _call_openai(messages)
            decision = _extract_json(text)
            _log_decision(context, decision)
            _cache_store(resolved_cache_key, fingerprint, decision)
            return decision
        except Exception as e:
            logging.error(f"OpenAI call failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)

    return _hold_decision("AI error")


async def get_ai_trade_decisions_batch(
    contexts: list[dict],
    *,
    cache_key_prefix: str = "batch",
    min_interval_seconds: int | None = None,
    max_cache_age_seconds: int | None = None,
) -> dict[str, dict]:
    enabled = os.getenv("ENABLE_AI_STRATEGY", "true").lower() in ("true", "1", "yes")
    normalized_contexts = [context for context in contexts if isinstance(context, dict) and context]
    if not normalized_contexts:
        return {}
    if not enabled:
        return {
            str(context.get("symbol") or f"ctx-{index}").upper(): _hold_decision("AI disabled")
            for index, context in enumerate(normalized_contexts)
        }

    _refresh_openai_api_key()
    if not openai.api_key:
        logging.error("OPENAI_API_KEY not set")
        return {
            str(context.get("symbol") or f"ctx-{index}").upper(): _hold_decision("No API key")
            for index, context in enumerate(normalized_contexts)
        }

    min_interval = max(0, int(min_interval_seconds if min_interval_seconds is not None else _default_min_interval()))
    cache_ttl = max(min_interval, int(max_cache_age_seconds if max_cache_age_seconds is not None else _default_cache_ttl()))
    results: dict[str, dict] = {}
    pending: list[tuple[str, str, str, dict]] = []

    for index, context in enumerate(normalized_contexts):
        symbol = str(context.get("symbol") or f"CTX-{index}").strip().upper()
        resolved_cache_key = f"{cache_key_prefix}:{_cache_key_for_context(context, symbol)}"
        fingerprint = _context_hash(context)
        cached = _cache_hit(resolved_cache_key, fingerprint, min_interval, cache_ttl)
        if cached:
            results[symbol] = cached
        else:
            pending.append((symbol, resolved_cache_key, fingerprint, context))

    if not pending:
        return results

    user_payload = {"contexts": [context for _, _, _, context in pending], "instruction": BATCH_RETURN_INSTRUCTION}
    messages = [
        {
            "role": "system",
            "content": (
                "You are a trading strategist assistant. Analyze each symbol context independently "
                "and return JSON with one decision per symbol."
            ),
        },
        {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
    ]

    parsed: dict[str, dict] = {}
    for attempt in range(_max_retries()):
        try:
            text = await _call_openai(messages)
            payload = _extract_json(text)
            for row in payload.get("decisions", []) if isinstance(payload, dict) else []:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "").strip().upper()
                if symbol:
                    parsed[symbol] = row
            break
        except Exception as e:
            logging.error(f"OpenAI batch call failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)

    for symbol, resolved_cache_key, fingerprint, context in pending:
        decision = parsed.get(symbol) or _hold_decision("AI batch error")
        _log_decision(context, decision)
        _cache_store(resolved_cache_key, fingerprint, decision)
        results[symbol] = decision

    return results


def get_last_decision(log_path: str = "logs/ai_decisions.log") -> dict:
    """Return the most recent AI decision and context from log file."""
    path = Path(log_path)
    if not path.is_file():
        return {}
    try:
        lines = path.read_text().strip().splitlines()
        if not lines:
            return {}
        last = lines[-1]
        ts_part, rest = last.split(" ", 1)
        ctx_str = rest.split("context=")[1].split(" decision=")[0]
        dec_str = rest.split("decision=")[1]
        return {
            "timestamp": ts_part,
            "context": json.loads(ctx_str),
            "decision": json.loads(dec_str),
        }
    except Exception as e:
        logging.error(f"Failed to read last AI decision: {e}")
        return {}


async def _fetch_news_headlines(api_key: str, limit: int = 20) -> list[str]:
    """Return a list of recent business/crypto news headlines."""
    import aiohttp

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "pageSize": limit,
        "apiKey": api_key,
    }
    headlines: list[str] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                for art in data.get("articles", []):
                    title = art.get("title")
                    if title:
                        headlines.append(title)
    except Exception as e:
        logging.error(f"NewsAPI fetch failed: {e}")
    return headlines


async def ai_discover_assets(base_symbols: list[str] | None = None) -> list[str]:
    """Return 2-3 trending symbols not already in base_symbols."""
    enabled = os.getenv("ENABLE_AI_ASSET_DISCOVERY", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    if not enabled:
        return []

    _refresh_openai_api_key()
    if not openai.api_key:
        logging.error("OPENAI_API_KEY not set for asset discovery")
        return []

    base_symbols = base_symbols or []
    news_key = os.getenv("NEWSAPI_KEY")
    headlines: list[str] = []
    if news_key:
        headlines = await _fetch_news_headlines(news_key, limit=20)

    user_msg = (
        "Existing symbols: "
        + ",".join(base_symbols)
        + "\nRecent headlines:\n"
        + "\n".join(headlines[:20])
        + "\nSuggest up to 3 additional liquid trading symbols not already in the list."
        " Return JSON: { symbols: [symbol,...], reason }"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a market analyst recommending assets.",
        },
        {"role": "user", "content": user_msg},
    ]

    try:
        text = await _call_openai(messages)
        data = _extract_json(text)
        symbols = [s.strip().upper() for s in data.get("symbols", [])]
        reason = data.get("reason", "")
        symbols = [s for s in symbols if s and s not in base_symbols]
        if symbols:
            logging.info(f"[AI_DISCOVERED] {symbols} reason={reason}")
        return symbols
    except Exception as e:
        logging.error(f"AI asset discovery failed: {e}")
        return []


async def get_conviction_score(context: dict) -> float:
    """Return a conviction score between 0 and 1 using GPT if available."""
    try:
        decision = await get_ai_trade_decision_cached(context)
        return float(decision.get("confidence", 0.5))
    except Exception:
        return 0.5
