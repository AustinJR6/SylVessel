# api/forex_api.py

import logging
from urllib.parse import urljoin
import aiohttp
from api.base_api import BaseAPI
from utils.guardrails import log_live_trade

class ForexAPI(BaseAPI):
    """
    Forex API client (e.g. OANDA). Subclasses BaseAPI for HTTP logic.
    """

    def __init__(self, api_key: str, account_id: str, base_url: str = "https://api-fxpractice.oanda.com", simulation_mode: bool = True, portfolio=None, config: dict | None = None, trade_cooldown: int = 30):
        super().__init__(base_url)
        self.api_key = api_key
        self.account_id = account_id
        self.simulation_mode = simulation_mode
        self.portfolio = portfolio
        self.config = config or {}
        self.trade_cooldown = trade_cooldown
        self._last_trade: dict[str, float] = {}
        # Set auth header for all requests
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    async def get_account_info(self) -> dict:
        if self.simulation_mode:
            logging.debug("ForexAPI: simulation mode – returning mock account info")
            if self.portfolio:
                return self.portfolio.account_snapshot()
            return {"balance": 100000.0}
        path = f"/v3/accounts/{self.account_id}"
        return await self.get(path)

    async def fetch_price(self, instrument: str) -> dict:
        """
        Fetch the latest bid/ask for a given Forex instrument.
        """
        if self.simulation_mode:
            logging.debug(f"ForexAPI: simulation mode – returning mock price for {instrument}")
            return {"instrument": instrument, "bid": 1.2345, "ask": 1.2348}
        path = f"/v3/accounts/{self.account_id}/pricing?instruments={instrument}"
        return await self.get(path)

    async def place_order(self, instrument: str, units: float, order_type: str = "MARKET", price: float = None) -> dict:
        """
        Place a market or limit order.
        """
        if not self.simulation_mode:
            now = aiohttp.helpers.utcnow().timestamp()
            last = self._last_trade.get(instrument)
            if last and now - last < self.trade_cooldown:
                logging.warning(f"Duplicate trade blocked for {instrument}")
                return {"status": "blocked", "reason": "duplicate"}
            self._last_trade[instrument] = now

        if self.simulation_mode:
            logging.info(f"ForexAPI: simulation order for {units} units of {instrument}")
            trade_price = price
            if trade_price is None:
                data = await self.fetch_price(instrument)
                trade_price = float(data.get("bid") or 0)
            if self.portfolio:
                execution = self.portfolio.execute_trade(instrument, "buy" if units > 0 else "sell", abs(units), trade_price, 0.0)
                if execution.get("status") != "filled":
                    return execution
            return {"status": "simulated", "instrument": instrument, "units": units, "price": trade_price}
        path = f"/v3/accounts/{self.account_id}/orders"
        body = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": order_type,
            }
        }
        if price is not None:
            body["order"]["price"] = str(price)
        result = await self.post(path, body)
        trade_price = price if price is not None else 0.0
        if trade_price == 0.0:
            try:
                data = await self.fetch_price(instrument)
                trade_price = float(data.get("bid") or 0)
            except Exception:
                trade_price = 0.0
        await log_live_trade(
            instrument,
            "buy" if units > 0 else "sell",
            abs(units),
            trade_price,
            self.config,
            market="forex",
            confidence=None,
            risk_pct=self.config.get("forex_settings", {}).get("risk_per_trade") * 100 if self.config.get("forex_settings") else None,
        )
        return result

    async def close(self):
        await super().close()
