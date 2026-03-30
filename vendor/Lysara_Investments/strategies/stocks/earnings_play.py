# strategies/stocks/earnings_play.py

import asyncio
import logging
from strategies.base_strategy import BaseStrategy

class EarningsPlayStrategy(BaseStrategy):
    def __init__(self, api, risk, config, db, symbol_list):
        super().__init__(api, risk, config, db, symbol_list)
        self.interval = 60  # Check every 60 seconds

    async def run(self):
        while True:
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    price_data = await self.api.fetch_market_price(symbol)
                    price = float(price_data.get("price") or price_data.get("last_trade_price", 0))
                    # Placeholder logic for earnings flag
                    earnings_soon = self.mock_earnings_event(symbol)

                    if earnings_soon:
                        await self.enter_trade(symbol, price, "buy")

                except Exception as e:
                    logging.error(f"[EarningsPlay] Error for {symbol}: {e}")

            await asyncio.sleep(self.interval)

    def mock_earnings_event(self, symbol):
        """Stub for earnings calendar integration."""
        # Replace this with actual API-based detection in future
        return symbol.endswith("L")  # dumb logic to simulate

    async def enter_trade(self, symbol, price, side):
        if not self.trading_enabled(symbol):
            logging.info("Strategy or symbol disabled. Trade skipped.")
            return
        if not await self.risk.check_daily_loss():
            logging.warning("Daily loss limit reached. Trade blocked.")
            return
        qty = self.risk.get_position_size(price)
        if qty <= 0:
            logging.warning(f"EarningsPlay: invalid position size for {symbol}")
            return

        await self.api.place_order(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type="market",
            price=price,
            confidence=0.0,
        )

        self.db.log_trade(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            reason="earnings_play",
            market="stocks"
        )

        logging.info(f"{side.upper()} {symbol} earnings play triggered at {price}")
