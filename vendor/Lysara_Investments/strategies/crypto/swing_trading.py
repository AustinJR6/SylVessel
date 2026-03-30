from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from data.market_data_daily import get_daily_history, refresh_symbol_history
from services.position_registry import RegisteredPosition, get_registry
from strategies.base_strategy import BaseStrategy


class SwingTradingStrategy(BaseStrategy):
    def __init__(self, api, risk, config, db, symbol_list):
        super().__init__(api, risk, config, db, symbol_list)
        self.interval = int(config.get("swing_review_interval_seconds", 4 * 3600))
        self.min_hold_seconds = int(config.get("swing_min_hold_seconds", 24 * 3600))
        self.stop_loss_pct = float(config.get("swing_stop_loss_pct", 0.06))
        self.take_profit_pct = float(config.get("swing_take_profit_pct", 0.12))
        self.open_trade_plan: dict[str, dict] = {}

    async def run(self):
        while True:
            for symbol in self.symbols:
                try:
                    if not self.trading_enabled(symbol):
                        continue
                    rows = get_daily_history(symbol, limit=120)
                    if len(rows) < 40:
                        rows = await refresh_symbol_history(symbol, range_label="1y")
                    closes = [float(row.get("close") or 0.0) for row in rows if float(row.get("close") or 0.0) > 0]
                    if len(closes) < 40:
                        self.record_decision(symbol, "observe", "skipped", ["insufficient_daily_history"], market="crypto")
                        continue
                    price = closes[-1]
                    if await self._manage_open_position(symbol, price):
                        continue
                    fast = sum(closes[-10:]) / 10.0
                    slow = sum(closes[-30:]) / 30.0
                    if fast > slow * 1.01:
                        await self.enter_trade(symbol, price, "buy")
                    else:
                        self.record_decision(symbol, "observe", "skipped", ["trend_not_strong_enough"], market="crypto")
                except Exception as exc:
                    logging.error("[SwingTrading] Error on %s: %s", symbol, exc)
                    self.record_decision(symbol, "observe", "error", [str(exc)], market="crypto")
            await asyncio.sleep(max(60, self.interval))

    async def enter_trade(self, symbol: str, price: float, side: str):
        if not self.trading_enabled(symbol):
            self.record_decision(symbol, side, "blocked", ["strategy_or_symbol_disabled"], market="crypto")
            return False
        if side == "sell":
            current_qty = await self._current_position_qty(symbol)
            if current_qty <= 0:
                self.record_decision(symbol, side, "blocked", ["no_open_position"], market="crypto")
                return False
            qty = current_qty
        else:
            if not await self.risk.check_daily_loss():
                self.record_decision(symbol, side, "blocked", ["daily_loss_limit"], market="crypto")
                return False
            qty = self.risk.get_position_size(price)
            if qty <= 0:
                self.record_decision(symbol, side, "blocked", ["invalid_position_size"], market="crypto")
                return False
        await self.api.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="MARKET",
            price=price,
            confidence=0.65,
        )
        self.db.log_trade(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            profit_loss=None,
            reason="swing_trading",
            market="crypto",
        )
        registry = self.position_registry or get_registry()
        if side == "buy":
            self.open_trade_plan[symbol] = {
                "entry_price": float(price),
                "quantity": float(qty),
                "opened_at": datetime.utcnow().isoformat(),
                "stop_price": float(price) * (1 - self.stop_loss_pct),
                "target_price": float(price) * (1 + self.take_profit_pct),
            }
            registry.register(
                RegisteredPosition(
                    symbol=symbol,
                    strategy_name=self.__class__.__name__,
                    side="buy",
                    entry_price=float(price),
                    quantity=float(qty),
                    stop_price=self.open_trade_plan[symbol]["stop_price"],
                    target_price=self.open_trade_plan[symbol]["target_price"],
                )
            )
        else:
            self.open_trade_plan.pop(symbol, None)
            registry.release(symbol, self.__class__.__name__)
        self.record_decision(symbol, side, "executed", ["swing_strategy"], market="crypto")
        return True

    async def _manage_open_position(self, symbol: str, price: float) -> bool:
        plan = self.open_trade_plan.get(symbol)
        if not plan:
            return False
        opened_at = _parse_ts(plan.get("opened_at"))
        if opened_at and datetime.utcnow() - opened_at < timedelta(seconds=self.min_hold_seconds):
            self.record_decision(symbol, "hold", "skipped", ["minimum_hold_active"], market="crypto")
            return True
        if price >= float(plan.get("target_price") or 0.0):
            await self.enter_trade(symbol, price, "sell")
            return True
        if price <= float(plan.get("stop_price") or 0.0):
            await self.enter_trade(symbol, price, "sell")
            return True
        return False

    async def _current_position_qty(self, symbol: str) -> float:
        holdings = await self.api.fetch_holdings()
        if not isinstance(holdings, dict):
            return 0.0
        asset_code = symbol.split("-", 1)[0].upper()
        return float(holdings.get(symbol, holdings.get(asset_code, 0.0)) or 0.0)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None
