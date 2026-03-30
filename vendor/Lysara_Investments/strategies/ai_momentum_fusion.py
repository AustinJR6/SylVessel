from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from services.ai_strategist import get_ai_trade_decisions_batch
from services.position_registry import RegisteredPosition, get_registry
from strategies.base_strategy import BaseStrategy


class AIMomentumFusion(BaseStrategy):
    """Momentum strategy with a stronger AI review gate."""

    def __init__(self, api, risk, config, db, symbol_list, **kwargs):
        super().__init__(api, risk, config, db, symbol_list)
        self.interval = int(config.get("market_poll_interval_seconds", 20))
        self.ai_min_confidence = float(config.get("min_conviction_score", config.get("ai_min_confidence", 0.72)))
        self.ai_review_interval_seconds = int(config.get("ai_review_interval_seconds", 300))
        self.ai_decision_cache_ttl_seconds = int(config.get("ai_decision_cache_ttl_seconds", 1800))
        self.last_ai_review_at: dict[str, datetime] = {}
        self.position_registry = kwargs.get("position_registry")

    async def run(self):
        while True:
            pending_reviews: list[tuple[str, float, dict]] = []
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    market_data = await self.api.fetch_market_price(symbol)
                    price = float(market_data.get("price") or 0.0)
                    if price <= 0:
                        self.record_decision(symbol, "observe", "blocked", ["missing_market_price"], market="crypto")
                        continue
                    self.price_history[symbol].append(price)
                    self.price_history[symbol] = self.price_history[symbol][-120:]
                    if not self._ai_review_due(symbol):
                        continue
                    pending_reviews.append((symbol, price, self._build_context(symbol, price)))
                except Exception as exc:
                    logging.error("[AIMomentumFusion] Error on %s: %s", symbol, exc)
                    self.record_decision(symbol, "observe", "error", [str(exc)], market="crypto")

            if pending_reviews:
                decisions = await get_ai_trade_decisions_batch(
                    [context for _, _, context in pending_reviews],
                    cache_key_prefix="crypto:ai_momentum_fusion",
                    min_interval_seconds=self.ai_review_interval_seconds,
                    max_cache_age_seconds=self.ai_decision_cache_ttl_seconds,
                )
                for symbol, price, _ in pending_reviews:
                    self.last_ai_review_at[symbol] = datetime.utcnow()
                    decision = decisions.get(symbol, {"action": "hold", "confidence": 0.5, "reason": "no_ai_decision"})
                    action = str(decision.get("action") or "hold").strip().lower()
                    confidence = float(decision.get("confidence") or 0.0)
                    if action not in {"buy", "sell"}:
                        self.record_decision(symbol, "observe", "skipped", [str(decision.get("reason") or "hold")], confidence=confidence, market="crypto")
                        continue
                    if confidence < self.ai_min_confidence:
                        self.record_decision(symbol, action, "blocked", ["confidence_below_threshold"], confidence=confidence, market="crypto")
                        continue
                    await self.enter_trade(symbol, price, action, confidence, str(decision.get("reason") or "ai_momentum_fusion"))
            await asyncio.sleep(max(5, self.interval))

    async def enter_trade(self, symbol: str, price: float, side: str, confidence: float, reason: str = "ai_momentum_fusion"):
        if not self.trading_enabled(symbol):
            self.record_decision(symbol, side, "blocked", ["strategy_or_symbol_disabled"], confidence=confidence, market="crypto")
            return False
        allowed, reasons = self.registry_allows(symbol, side)
        if not allowed:
            self.record_decision(symbol, side, "blocked", reasons, confidence=confidence, market="crypto")
            return False
        if not await self.risk.check_daily_loss():
            self.record_decision(symbol, side, "blocked", ["daily_loss_limit"], confidence=confidence, market="crypto")
            return False
        qty = self.risk.get_position_size(price)
        if qty <= 0:
            self.record_decision(symbol, side, "blocked", ["invalid_position_size"], confidence=confidence, market="crypto")
            return False
        await self.api.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="MARKET",
            price=price,
            confidence=confidence,
        )
        self.db.log_trade(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            profit_loss=None,
            reason=reason,
            market="crypto",
        )
        registry = self.position_registry or get_registry()
        if side == "buy":
            registry.register(
                RegisteredPosition(
                    symbol=symbol,
                    strategy_name=self.__class__.__name__,
                    side="buy",
                    entry_price=float(price),
                    quantity=float(qty),
                )
            )
        else:
            registry.release(symbol, self.__class__.__name__)
        self.record_decision(symbol, side, "executed", [reason], confidence=confidence, market="crypto")
        return True

    def _build_context(self, symbol: str, price: float) -> dict:
        prices = self.price_history[symbol][-30:]
        momentum = 0.0
        if len(prices) >= 5 and prices[-5] > 0:
            momentum = (prices[-1] - prices[-5]) / prices[-5]
        return {
            "symbol": symbol,
            "market": "crypto",
            "price": price,
            "recent_momentum": round(momentum, 5),
            "recent_high": max(prices) if prices else price,
            "recent_low": min(prices) if prices else price,
            "position_status": "long" if self._position_quantity(symbol) > 0 else "flat",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _position_quantity(self, symbol: str) -> float:
        portfolio = getattr(self.api, "portfolio", None)
        if portfolio is None:
            return 0.0
        return float(getattr(portfolio, "open_positions", {}).get(symbol, 0.0) or 0.0)

    def _ai_review_due(self, symbol: str) -> bool:
        last_review = self.last_ai_review_at.get(symbol)
        if not last_review:
            return True
        return datetime.utcnow() - last_review >= timedelta(seconds=self.ai_review_interval_seconds)
