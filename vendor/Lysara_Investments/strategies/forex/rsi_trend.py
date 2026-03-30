import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from indicators.technical_indicators import relative_strength_index
from risk.risk import DynamicRisk
from services.ai_strategist import get_ai_trade_decisions_batch
from strategies.base_strategy import BaseStrategy

class ForexRSITrendStrategy(BaseStrategy):
    """Simple RSI-based trend following strategy for Forex."""

    def __init__(self, api, risk, config, db, symbol_list):
        super().__init__(api, risk, config, db, symbol_list)
        self.interval = int(config.get("market_poll_interval_seconds", 60))
        self.ai_min_confidence = float(config.get("ai_min_confidence", 0.7))
        self.ai_review_interval_seconds = int(config.get("ai_review_interval_seconds", 900))
        self.ai_decision_cache_ttl_seconds = int(
            config.get("ai_decision_cache_ttl_seconds", max(self.ai_review_interval_seconds * 2, 3600))
        )
        self.last_ai_review_at: dict[str, datetime] = {}
        self.dynamic_risk = DynamicRisk(
            risk,
            config.get("atr_period", 14),
            config.get("volatility_multiplier", 3),
        )

    @staticmethod
    def _extract_price(data: dict) -> float:
        """Handle both flattened and OANDA-native pricing payloads."""
        if not isinstance(data, dict):
            return 0.0
        if data.get("bid") is not None:
            try:
                return float(data.get("bid") or 0)
            except (TypeError, ValueError):
                return 0.0
        prices = data.get("prices")
        if isinstance(prices, list) and prices:
            first = prices[0] or {}
            bids = first.get("bids") or []
            if bids:
                try:
                    return float(bids[0].get("price") or 0)
                except (TypeError, ValueError, AttributeError):
                    return 0.0
        return 0.0

    async def run(self):
        while True:
            pending_reviews: list[tuple[str, float, dict]] = []
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    data = await self.api.fetch_price(symbol)
                    price = self._extract_price(data)
                    self.price_history[symbol].append(price)

                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]

                    if not self._ai_review_due(symbol):
                        continue
                    context = self._build_context(symbol, price)
                    pending_reviews.append((symbol, price, context))
                except Exception as e:
                    logging.error(f"[RSITrend] Error for {symbol}: {e}")

            if pending_reviews:
                decisions = await get_ai_trade_decisions_batch(
                    [context for _, _, context in pending_reviews],
                    cache_key_prefix="forex:rsi_trend",
                    min_interval_seconds=self.ai_review_interval_seconds,
                    max_cache_age_seconds=self.ai_decision_cache_ttl_seconds,
                )
                for symbol, price, _ in pending_reviews:
                    self.last_ai_review_at[symbol] = datetime.utcnow()
                    decision = decisions.get(symbol, {"action": "hold", "confidence": 0.5, "reason": "no_ai_decision"})

                    if (
                        decision.get("action") in ["buy", "sell"]
                        and decision.get("confidence", 0) >= self.ai_min_confidence
                    ):
                        await self.enter_trade(
                            symbol,
                            price,
                            decision.get("action"),
                            decision.get("confidence", 0.0),
                            decision.get("reason", "ai"),
                        )
                    else:
                        logging.info(f"AI decision skipped: {decision}")
            await asyncio.sleep(self.interval)

    def _build_context(self, symbol: str, price: float) -> dict:
        prices = self.price_history[symbol]
        vol = float(np.std(np.diff(prices[-10:]))) if len(prices) > 2 else 0.0
        trend = "sideways"
        if len(prices) >= 5:
            x = np.arange(5)
            y = np.array(prices[-5:])
            slope = np.polyfit(x, y, 1)[0]
            trend = "uptrend" if slope > 0 else "downtrend" if slope < 0 else "sideways"
        status = "flat"
        if getattr(self.api, "portfolio", None):
            qty = self.api.portfolio.open_positions.get(symbol, 0.0)
            if qty > 0:
                status = "long"
            elif qty < 0:
                status = "short"
        support = min(prices[-20:]) if prices else price
        resistance = max(prices[-20:]) if prices else price
        return {
            "symbol": symbol,
            "price": price,
            "volatility": round(vol, 6),
            "sentiment": 0.0,
            "position_status": status,
            "recent_trend": trend,
            "support": support,
            "resistance": resistance,
            "drawdown": self.risk.daily_loss,
            "loss_streak": self.risk.consec_losses,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def enter_trade(self, symbol, price, side, confidence, reason):
        if not self.trading_enabled(symbol):
            logging.info("Strategy or symbol disabled. Trade skipped.")
            return
        if not self.config.get("ENABLE_AI_TRADE_EXECUTION", False):
            logging.info("AI trade execution disabled. Trade skipped.")
            return
        qty = self.dynamic_risk.position_size(price, confidence, self.price_history[symbol])
        if qty <= 0:
            logging.warning(f"RSITrend: invalid position size for {symbol}")
            return

        if not self.config.get("simulation_mode", True) and not self.config.get("LIVE_TRADING_ENABLED", True):
            logging.info("Live trading disabled. Trade skipped.")
            return

        await self.api.place_order(
            instrument=symbol,
            units=qty if side == "buy" else -qty,
            order_type="MARKET",
            price=price,
            confidence=confidence,
        )

        self.db.log_trade(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            reason=reason,
            market="forex",
        )

        logging.info(
            f"{side.upper()} {symbol} @ {price} conf={confidence} reason={reason} size={qty}"
        )

    def _ai_review_due(self, symbol: str) -> bool:
        last_review = self.last_ai_review_at.get(symbol)
        if not last_review:
            return True
        return datetime.utcnow() - last_review >= timedelta(seconds=self.ai_review_interval_seconds)
