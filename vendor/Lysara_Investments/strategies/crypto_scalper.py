from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from indicators.technical_indicators import relative_strength_index
from services.position_registry import RegisteredPosition, get_registry
from strategies.base_strategy import BaseStrategy


class CryptoScalper(BaseStrategy):
    """Fast crypto scalper with simple RSI entry/exit rules."""

    def __init__(self, api, risk, config, db, symbol_list, **kwargs):
        super().__init__(api, risk, config, db, symbol_list)
        self.interval = int(config.get("scalp_interval_seconds", 10))
        self.rsi_period = int(config.get("scalp_rsi_period", 14))
        self.buy_threshold = float(config.get("scalp_rsi_buy_threshold", 35))
        self.sell_threshold = float(config.get("scalp_rsi_sell_threshold", 65))
        self.profit_target = float(config.get("scalp_profit_target_pct", 0.75)) / 100.0
        self.max_hold_ticks = int(config.get("scalp_timeout_ticks", 30))
        self.open_positions: dict[str, dict] = {}
        self.position_registry = kwargs.get("position_registry")

    async def run(self):
        while True:
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    data = await self.api.fetch_market_price(symbol)
                    price = float(data.get("price") or 0.0)
                    if price <= 0:
                        self.record_decision(symbol, "observe", "blocked", ["missing_market_price"], market="crypto")
                        continue
                    self.price_history[symbol].append(price)
                    self.price_history[symbol] = self.price_history[symbol][- max(self.rsi_period + 5, 40):]
                    if await self._manage_open_position(symbol, price):
                        continue
                    if len(self.price_history[symbol]) < self.rsi_period + 1:
                        continue
                    rsi = relative_strength_index(self.price_history[symbol], self.rsi_period)
                    if rsi <= self.buy_threshold:
                        await self.enter_trade(symbol, price, "buy", reason=f"rsi_{round(rsi, 2)}")
                    else:
                        self.record_decision(symbol, "observe", "skipped", [f"rsi_{round(rsi, 2)}"], market="crypto")
                except Exception as exc:
                    logging.error("[CryptoScalper] Error on %s: %s", symbol, exc)
                    self.record_decision(symbol, "observe", "error", [str(exc)], market="crypto")
            await asyncio.sleep(max(2, self.interval))

    async def enter_trade(self, symbol: str, price: float, side: str, reason: str = "scalper"):
        if not self.trading_enabled(symbol):
            self.record_decision(symbol, side, "blocked", ["strategy_or_symbol_disabled"], market="crypto")
            return False
        if side == "buy":
            allowed, reasons = self.registry_allows(symbol, side)
            if not allowed:
                self.record_decision(symbol, side, "blocked", reasons, market="crypto")
                return False
            if not await self.risk.check_daily_loss():
                self.record_decision(symbol, side, "blocked", ["daily_loss_limit"], market="crypto")
                return False
            qty = self.risk.get_position_size(price)
            if qty <= 0:
                self.record_decision(symbol, side, "blocked", ["invalid_position_size"], market="crypto")
                return False
            await self.api.place_order(symbol=symbol, side="buy", qty=qty, order_type="MARKET", price=price, confidence=0.0)
            self.open_positions[symbol] = {
                "entry_price": price,
                "quantity": qty,
                "ticks": 0,
            }
        else:
            current = self.open_positions.get(symbol)
            if not current:
                self.record_decision(symbol, side, "blocked", ["no_open_position"], market="crypto")
                return False
            qty = float(current.get("quantity") or 0.0)
            await self.api.place_order(symbol=symbol, side="sell", qty=qty, order_type="MARKET", price=price, confidence=0.0)
            self.open_positions.pop(symbol, None)
        self.db.log_trade(symbol=symbol, side=side, quantity=qty, price=price, profit_loss=None, reason=reason, market="crypto")
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
        self.record_decision(symbol, side, "executed", [reason], market="crypto")
        return True

    async def _manage_open_position(self, symbol: str, price: float) -> bool:
        current = self.open_positions.get(symbol)
        if not current:
            return False
        current["ticks"] = int(current.get("ticks") or 0) + 1
        entry_price = float(current.get("entry_price") or price)
        if price >= entry_price * (1 + self.profit_target):
            await self.enter_trade(symbol, price, "sell", reason="profit_target")
            return True
        if current["ticks"] >= self.max_hold_ticks:
            await self.enter_trade(symbol, price, "sell", reason="timeout_exit")
            return True
        if len(self.price_history[symbol]) >= self.rsi_period + 1:
            rsi = relative_strength_index(self.price_history[symbol], self.rsi_period)
            if rsi >= self.sell_threshold:
                await self.enter_trade(symbol, price, "sell", reason=f"rsi_exit_{round(rsi, 2)}")
                return True
        self.record_decision(symbol, "hold", "skipped", ["position_open"], market="crypto")
        return True
