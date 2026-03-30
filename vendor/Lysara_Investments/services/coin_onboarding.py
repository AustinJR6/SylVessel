from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from data.market_data_daily import get_daily_history, refresh_symbol_history
from services.runtime_store import get_runtime_store


@dataclass
class CoinOnboardingResult:
    symbol: str
    market: str
    strategy_key: str
    profile: dict[str, Any]
    activated: bool
    activation_reason: str


async def onboard_coin(symbol: str, force_strategy: str | None = None) -> CoinOnboardingResult:
    clean_symbol = str(symbol or "").strip().upper()
    rows = get_daily_history(clean_symbol, limit=120)
    if len(rows) < 40:
        rows = await refresh_symbol_history(clean_symbol, range_label="1y")
    closes = [float(row.get("close") or 0.0) for row in rows if float(row.get("close") or 0.0) > 0]
    if len(closes) < 20:
        profile = {
            "market": "crypto",
            "history_points": len(closes),
            "eligible": False,
            "reason": "insufficient_history",
        }
        get_runtime_store().save_symbol_profile(clean_symbol, profile)
        return CoinOnboardingResult(
            symbol=clean_symbol,
            market="crypto",
            strategy_key=str(force_strategy or "watch_only"),
            profile=profile,
            activated=False,
            activation_reason="insufficient_history",
        )

    volatility = _annualized_volatility(closes)
    trend_strength = _trend_strength(closes)
    mean_reversion_score = _mean_reversion_score(closes)
    if force_strategy:
        strategy_key = str(force_strategy).strip()
        activation_reason = "forced_strategy"
    elif volatility >= 0.55 and trend_strength >= 0.2:
        strategy_key = "swing"
        activation_reason = "high_volatility_trend"
    elif mean_reversion_score >= 0.65:
        strategy_key = "mean_reversion"
        activation_reason = "mean_reversion_bias"
    elif volatility >= 0.9:
        strategy_key = "crypto_scalper"
        activation_reason = "high_intraday_volatility"
    else:
        strategy_key = "momentum"
        activation_reason = "default_momentum"

    profile = {
        "market": "crypto",
        "history_points": len(closes),
        "volatility": round(volatility, 4),
        "trend_strength": round(trend_strength, 4),
        "mean_reversion_score": round(mean_reversion_score, 4),
        "recommended_strategy": strategy_key,
        "eligible": True,
        "updated_at": _utcnow_iso(),
    }
    get_runtime_store().save_symbol_profile(clean_symbol, profile)
    return CoinOnboardingResult(
        symbol=clean_symbol,
        market="crypto",
        strategy_key=strategy_key,
        profile=profile,
        activated=bool(force_strategy),
        activation_reason=activation_reason,
    )


def _annualized_volatility(closes: list[float]) -> float:
    returns = []
    for idx in range(1, len(closes)):
        previous = closes[idx - 1]
        current = closes[idx]
        if previous <= 0:
            continue
        returns.append((current - previous) / previous)
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / max(1, len(returns))
    return variance ** 0.5 * (365 ** 0.5)


def _trend_strength(closes: list[float]) -> float:
    if len(closes) < 20 or closes[0] <= 0:
        return 0.0
    return (closes[-1] - closes[-20]) / closes[-20]


def _mean_reversion_score(closes: list[float]) -> float:
    if len(closes) < 20:
        return 0.0
    mean_close = sum(closes[-20:]) / 20.0
    deviations = [abs(close - mean_close) / max(mean_close, 1e-9) for close in closes[-20:]]
    avg_dev = sum(deviations) / len(deviations)
    score = max(0.0, min(1.0, 1.0 - avg_dev * 5.0))
    return score


def _utcnow_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()
