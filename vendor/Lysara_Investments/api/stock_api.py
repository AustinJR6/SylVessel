# ==============================
# api/stock_api.py
# ==============================

import logging
from urllib.parse import urljoin
import aiohttp
import asyncio
from api.base_api import BaseAPI
from utils.guardrails import log_live_trade

class StockAPI(BaseAPI):
    """
    Stock trading API client (e.g., Robinhood). Subclasses BaseAPI for HTTP logic.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str = None,
        base_url: str = "https://api.robinhood.com",
        simulation_mode: bool = True,
        portfolio=None,
        config: dict | None = None,
        trade_cooldown: int = 30,
    ):
        super().__init__(base_url)
        self.api_key = api_key
        self.api_secret = api_secret
        self.simulation_mode = simulation_mode
        self.portfolio = portfolio
        self.config = config or {}
        self.trade_cooldown = trade_cooldown
        self._last_trade: dict[str, float] = {}
        # For Robinhood, token auth might go here
        if not simulation_mode:
            # Example header for real-world usage
            self.session.headers.update({
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            })

    async def fetch_account_info(self) -> dict:
        """
        Return account balances or mock in simulation.
        """
        if self.simulation_mode:
            logging.debug("StockAPI: simulation mode – returning mock account info")
            if self.portfolio:
                return self.portfolio.account_snapshot()
            return {"cash": 10000.0, "portfolio_value": 10000.0}
        path = "/accounts/"
        return await self.get(path)

    async def fetch_holdings(self) -> dict:
        """
        Return open positions or mock in simulation.
        """
        if self.simulation_mode:
            logging.debug("StockAPI: simulation mode – returning mock holdings")
            if self.portfolio:
                return self.portfolio.holdings_snapshot()
            return {}
        path = "/positions/"
        data = await self.get(path)
        positions = {}
        for item in data.get("results", []):
            symbol = item.get("instrument", "")
            qty = float(item.get("quantity", 0))
            positions[symbol] = qty
        return positions

    async def fetch_market_price(self, symbol: str) -> dict:
        """
        Get latest bid/ask or mock.
        """
        if self.simulation_mode:
            logging.debug(f"StockAPI: sim price for {symbol}")
            return {"symbol": symbol, "bid": 0.0, "ask": 0.0, "last_trade_price": 0.0}
        path = f"/quotes/?symbols={symbol}"
        return await self.get(path)

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float = None,
        order_type: str = "market",
        **kwargs,
    ) -> dict:
        """
        Place market or limit order; returns order details or mock.
        """
        if not self.simulation_mode:
            now = asyncio.get_event_loop().time()
            last = self._last_trade.get(symbol)
            if last and now - last < self.trade_cooldown:
                logging.warning(f"Duplicate trade blocked for {symbol}")
                return {"status": "blocked", "reason": "duplicate"}
            self._last_trade[symbol] = now

        if self.simulation_mode:
            logging.info(f"StockAPI: sim {order_type} order {side} {quantity} {symbol}")
            trade_price = price
            if trade_price is None:
                data = await self.fetch_market_price(symbol)
                trade_price = float(data.get("price") or data.get("last_trade_price", 0))
            if self.portfolio:
                conf = kwargs.get("confidence", 0.0)
                execution = self.portfolio.execute_trade(symbol, side, quantity, trade_price, conf)
                if execution.get("status") != "filled":
                    return {"id": "sim_order", **execution}
            return {"id": "sim_order", "status": "filled", "symbol": symbol, "side": side, "quantity": quantity, "price": trade_price}

        path = "/orders/"
        body = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
        }
        if order_type == "limit" and price is not None:
            body["price"] = price
        result = await self.post(path, body)
        trade_price = price if price is not None else 0.0
        if trade_price == 0.0:
            try:
                data = await self.fetch_market_price(symbol)
                trade_price = float(data.get("price") or data.get("last_trade_price", 0))
            except Exception:
                trade_price = 0.0
        await log_live_trade(
            symbol,
            side,
            quantity,
            trade_price,
            self.config,
            market="stock",
            confidence=kwargs.get("confidence"),
            risk_pct=self.config.get("stocks_settings", {}).get("risk_per_trade") * 100 if self.config.get("stocks_settings") else None,
        )
        return result

    async def close(self):
        """Clean up HTTP session."""
        await super().close()
