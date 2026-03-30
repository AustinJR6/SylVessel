# strategies/crypto/mean_reversion.py

import asyncio
import logging
from indicators.technical_indicators import moving_average
from services.position_registry import RegisteredPosition, get_registry
from strategies.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, api, risk, config, db, symbol_list, **kwargs):
        super().__init__(api, risk, config, db, symbol_list)
        self.interval = 10  # seconds
        self.position_registry = kwargs.get("position_registry")

    async def run(self):
        while True:
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    data = await self.api.fetch_market_price(symbol)
                    price = float(data.get("price", 0))
                    self.price_history[symbol].append(price)

                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]

                    ma = moving_average(self.price_history[symbol], 20)

                    if price < 0.98 * ma:
                        await self.enter_trade(symbol, price, "buy")
                    elif price > 1.02 * ma:
                        await self.enter_trade(symbol, price, "sell")

                except Exception as e:
                    logging.error(f"[MeanReversion] Error on {symbol}: {e}")

            await asyncio.sleep(self.interval)

    async def enter_trade(self, symbol, price, side):
        if not self.trading_enabled(symbol):
            logging.info("Strategy or symbol disabled. Trade skipped.")
            self.record_decision(symbol, side, "blocked", ["strategy_or_symbol_disabled"], market="crypto")
            return
        if side == "buy":
            allowed, reasons = self.registry_allows(symbol, side)
            if not allowed:
                self.record_decision(symbol, side, "blocked", reasons, market="crypto")
                return
        if not await self.risk.check_daily_loss():
            logging.warning("Daily loss limit reached. Trade blocked.")
            self.record_decision(symbol, side, "blocked", ["daily_loss_limit"], market="crypto")
            return
        qty = self.risk.get_position_size(price)
        if qty <= 0:
            logging.warning("Position size is zero or invalid.")
            self.record_decision(symbol, side, "blocked", ["invalid_position_size"], market="crypto")
            return

        order = await self.api.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_type="MARKET",
            confidence=0.0,
        )

        self.db.log_trade(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            profit_loss=None,
            reason="mean_reversion",
            market="crypto"
        )

        logging.info(f"Executed {side.upper()} {symbol} @ {price} [Mean Reversion]")
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
        self.record_decision(symbol, side, "executed", ["mean_reversion"], market="crypto")
