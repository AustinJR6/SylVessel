# strategies/stocks/stock_momentum.py

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np

from indicators.technical_indicators import moving_average
from services.ai_strategist import get_ai_trade_decisions_batch
from strategies.base_strategy import BaseStrategy

class StockMomentumStrategy(BaseStrategy):
    def __init__(self, api, risk, config, db, symbol_list):
        super().__init__(api, risk, config, db, symbol_list)
        self.ma_period = config.get("moving_average_period", 20)
        self.interval = int(config.get("market_poll_interval_seconds", 60))
        self.ai_min_confidence = float(config.get("ai_min_confidence", 0.7))
        self.ai_review_interval_seconds = int(config.get("ai_review_interval_seconds", 900))
        self.ai_decision_cache_ttl_seconds = int(
            config.get("ai_decision_cache_ttl_seconds", max(self.ai_review_interval_seconds * 2, 3600))
        )
        self.last_ai_review_at: dict[str, datetime] = {}

    async def run(self):
        while True:
            pending_reviews: list[tuple[str, float, dict]] = []
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    price_data = await self.api.fetch_market_price(symbol)
                    price = float(price_data.get("price") or price_data.get("last_trade_price", 0))
                    self.price_history[symbol].append(price)

                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]

                    if not self._ai_review_due(symbol):
                        continue
                    context = self._build_context(symbol, price)
                    pending_reviews.append((symbol, price, context))

                except Exception as e:
                    logging.error(f"[StockMomentum] Error for {symbol}: {e}")

            if pending_reviews:
                decisions = await get_ai_trade_decisions_batch(
                    [context for _, _, context in pending_reviews],
                    cache_key_prefix="stocks:momentum",
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

    async def enter_trade(self, symbol, price, side, confidence, reason="ai"):
        if not self.trading_enabled(symbol):
            logging.info("Strategy or symbol disabled. Trade skipped.")
            return
        if not self.config.get("ENABLE_AI_TRADE_EXECUTION", False):
            logging.info("AI trade execution disabled. Trade skipped.")
            return
        if not await self.risk.check_daily_loss():
            logging.warning("Daily loss limit reached. Trade blocked.")
            return
        qty = self.risk.get_position_size(price)
        if qty <= 0:
            logging.warning(f"StockMomentum: invalid position size for {symbol}")
            return

        if not self.config.get("simulation_mode", True) and not self.config.get("LIVE_TRADING_ENABLED", True):
            logging.info("Live trading disabled. Trade skipped.")
            return

        await self.api.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            type="market",
            price=price,
            confidence=confidence,
        )

        self.db.log_trade(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            reason=reason,
            market="stocks"
        )

        logging.info(
            f"{side.upper()} {symbol} @ {price} conf={confidence} via ai strategist"
        )

    def _ai_review_due(self, symbol: str) -> bool:
        last_review = self.last_ai_review_at.get(symbol)
        if not last_review:
            return True
        return datetime.utcnow() - last_review >= timedelta(seconds=self.ai_review_interval_seconds)
