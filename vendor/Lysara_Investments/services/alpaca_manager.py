import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from data.price_cache import get_price
from utils.guardrails import log_live_trade
from alpaca_client import (
    get_account,
    get_positions,
    place_order as api_place_order,
    fetch_market_price,
)

load_dotenv()


class AlpacaManager:
    """Thin wrapper around Alpaca REST API using `requests`."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
        simulation_mode: bool = True,
        portfolio=None,
        config: Optional[dict] = None,
        trade_cooldown: int = 30,
    ) -> None:
        self.simulation_mode = simulation_mode
        self.portfolio = portfolio
        self.config = config or {}
        self.trade_cooldown = trade_cooldown
        self._last_trade: dict[str, float] = {}

        # Normalize base URL to avoid accidental double "/v2" pathing.
        normalized_base_url = (base_url or "https://paper-api.alpaca.markets").rstrip("/")
        if normalized_base_url.endswith("/v2"):
            normalized_base_url = normalized_base_url[:-3]

        # ensure credentials are available to alpaca_client
        os.environ.setdefault("ALPACA_API_KEY", api_key)
        os.environ.setdefault("ALPACA_SECRET_KEY", api_secret)
        os.environ.setdefault("ALPACA_BASE_URL", normalized_base_url)
        self.base_url = normalized_base_url

    async def fetch_account_info(self) -> dict:
        """Return normalized account info for RiskManager compatibility."""
        data = await self.get_account()
        if not isinstance(data, dict):
            return {"balance": 0.0}
        equity = data.get("equity")
        if equity is None:
            equity = data.get("portfolio_value")
        if equity is None:
            equity = data.get("cash", 0.0)
        try:
            return {
                "balance": float(equity or 0.0),
                "portfolio_value": float(data.get("portfolio_value", equity or 0.0)),
                "cash": float(data.get("cash", 0.0)),
            }
        except Exception:
            return {"balance": 0.0}

    async def get_account(self):
        if self.simulation_mode:
            logging.debug("AlpacaManager: returning mock account data")
            if self.portfolio:
                return self.portfolio.account_snapshot()
            return {"cash": 10000.0, "equity": 10000.0, "buying_power": 10000.0}
        return await get_account()

    async def get_positions(self):
        if self.simulation_mode:
            logging.debug("AlpacaManager: returning empty positions in sim mode")
            if self.portfolio:
                return [
                    {"symbol": symbol, "qty": quantity}
                    for symbol, quantity in self.portfolio.holdings_snapshot().items()
                ]
            return []
        return await get_positions()

    async def fetch_market_price(self, symbol: str) -> dict:
        if self.simulation_mode:
            cached = get_price(symbol)
            if cached:
                return cached
            logging.debug(f"No cached market price for {symbol} in simulation mode")
            return {"price": 0.0, "source": "simulation", "time": None}
        logging.debug(f"Fetching market price for {symbol} via Alpaca API")
        return await fetch_market_price(symbol)

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float | None = None,
        type: str = "market",
        time_in_force: str = "day",
        price: float | None = None,
        **kwargs,
    ):
        # Backward-compatible aliases used by strategy code.
        if qty is None:
            qty = kwargs.get("quantity")
        if "order_type" in kwargs and not kwargs.get("type"):
            type = kwargs["order_type"]
        if qty is None:
            raise ValueError("Missing order quantity")

        if self.simulation_mode:
            logging.info(f"[SIM] {side.upper()} {qty} {symbol}")
            if self.portfolio:
                if price is None:
                    price = (await self.fetch_market_price(symbol)).get("price", 0)
                execution = self.portfolio.execute_trade(symbol, side, qty, price)
                if execution.get("status") != "filled":
                    return {"id": "sim", **execution}
            return {"id": "sim", "status": "filled", "symbol": symbol, "qty": qty, "price": price}

        now = asyncio.get_event_loop().time()
        last = self._last_trade.get(symbol)
        if last and now - last < self.trade_cooldown:
            logging.warning(f"Duplicate trade blocked for {symbol}")
            return {"status": "blocked", "reason": "duplicate"}
        self._last_trade[symbol] = now

        order = await api_place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            type=type,
            time_in_force=time_in_force,
        )
        trade_price = price if price is not None else (
            await self.fetch_market_price(symbol)
        ).get("price", 0)
        await log_live_trade(
            symbol,
            side,
            qty,
            trade_price,
            self.config,
            market="stock",
            confidence=kwargs.get("confidence"),
            risk_pct=self.config.get("stocks_settings", {}).get("risk_per_trade") * 100 if self.config.get("stocks_settings") else None,
        )
        return order

