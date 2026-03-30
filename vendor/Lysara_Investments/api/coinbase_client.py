"""Deprecated Coinbase trading client.

This module previously handled trading via Coinbase. Binance is now the
primary exchange and this client remains only for archival purposes and
non-trading utilities. It should not be imported by active modules.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

try:
    from coinbase_agentkit.broker import BrokerAPI
except Exception:  # pragma: no cover - fallback if AgentKit missing
    BrokerAPI = None
    try:
        from coinbase_advanced_trade import AdvancedTradeClient as BrokerAPI
    except ModuleNotFoundError:  # pragma: no cover
        try:
            from coinbase.rest import RESTClient as BrokerAPI
        except ModuleNotFoundError:  # pragma: no cover - final fallback
            class BrokerAPI:
                """Minimal stub for tests when no SDKs are installed."""

                def __init__(self, *_, **__):
                    pass

from utils.guardrails import log_live_trade


class CoinbaseClient:
    """Thin async wrapper around Coinbase AgentKit's BrokerAPI."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        simulation_mode: bool = True,
        portfolio=None,
        config: Optional[Dict] = None,
        trade_cooldown: int = 30,
    ) -> None:
        self.simulation_mode = simulation_mode
        self.portfolio = portfolio
        self.config = config or {}
        self.trade_cooldown = trade_cooldown
        self._last_trade: Dict[str, float] = {}
        self._mock_equity = 10000.0
        self._mock_holdings: Dict[str, float] = {}

        self.client = None
        if not simulation_mode and BrokerAPI:
            self.client = BrokerAPI(api_key=api_key, api_secret=api_secret)
        elif not simulation_mode:
            logging.warning("BrokerAPI unavailable; Coinbase client disabled")

    async def _run(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def get_accounts(self) -> Any:
        """Return raw account data from Coinbase."""
        if self.simulation_mode or not self.client:
            return []
        return await self._run(self.client.get_accounts)

    async def get_balances(self) -> Dict[str, float]:
        """Return currency balances as a dict."""
        accounts = await self.get_accounts()
        holdings: Dict[str, float] = {}
        for acct in getattr(accounts, "accounts", []):
            cur = getattr(acct, "currency", None)
            bal = getattr(acct, "available_balance", None)
            if bal is not None:
                val = float(getattr(bal, "value", 0) or 0)
                if cur and val:
                    holdings[cur] = val
        logging.debug(f"Balances fetched: {holdings}")
        return holdings

    # ------------------------------------------------------------------
    # Account and holdings
    # ------------------------------------------------------------------
    async def get_account_value(self) -> float:
        info = await self.fetch_account_info()
        return float(info.get("balance", 0)) if isinstance(info, dict) else 0.0

    async def fetch_account_info(self) -> Dict[str, Any]:
        """Return USD balance from Coinbase or mock data."""
        if self.simulation_mode:
            logging.debug("CoinbaseClient: simulation_mode – returning mock account info")
            if self.portfolio:
                return self.portfolio.account_snapshot()
            return {"currency": "USD", "balance": self._mock_equity}
        for attempt in range(1, 4):
            try:
                accounts = await self.get_accounts()
                usd_balance = 0.0
                for acct in getattr(accounts, "accounts", []):
                    if getattr(acct, "currency", "") == "USD" and acct.available_balance:
                        bal = getattr(acct.available_balance, "value", 0) or 0
                        usd_balance = float(bal)
                        break
                logging.debug(f"Account info response: {usd_balance}")
                return {"currency": "USD", "balance": usd_balance}
            except Exception as e:
                logging.error(f"fetch_account_info failed (attempt {attempt}): {e}")
                await asyncio.sleep(attempt)
        return {"currency": "USD", "balance": 0.0}

    async def get_holdings(self) -> Dict[str, float]:
        if self.simulation_mode:
            logging.debug("CoinbaseClient: simulation_mode – returning mock holdings")
            if self.portfolio:
                return self.portfolio.holdings_snapshot()
            return self._mock_holdings
        for attempt in range(1, 4):
            try:
                holdings = await self.get_balances()
                return holdings
            except Exception as e:
                logging.error(f"get_holdings failed (attempt {attempt}): {e}")
                await asyncio.sleep(attempt)
        return {}

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    async def place_order(self, symbol: str, side: str, qty: float) -> Any:
        if self.simulation_mode:
            logging.info(f"CoinbaseClient SIM {side} {qty} {symbol}")
            price = 0.0
            if self.portfolio:
                price = (await self.fetch_market_price(symbol)).get("price", 0.0)
                execution = self.portfolio.execute_trade(symbol, side, qty, price)
                if execution.get("status") != "filled":
                    return {"id": "sim_order", **execution}
            return {"id": "sim_order", "status": "done", "filled_size": qty}

        now = asyncio.get_event_loop().time()
        last = self._last_trade.get(symbol)
        if last and now - last < self.trade_cooldown:
            logging.warning(f"Duplicate trade blocked for {symbol}")
            return {"status": "blocked"}
        self._last_trade[symbol] = now

        for attempt in range(1, 4):
            try:
                client_id = uuid.uuid4().hex
                if side.lower() == "buy":
                    result = await self._run(
                        self.client.market_order_buy, client_id, symbol, base_size=str(qty)
                    )
                else:
                    result = await self._run(
                        self.client.market_order_sell, client_id, symbol, base_size=str(qty)
                    )
                logging.debug(f"Order response: {result}")
                price = (await self.fetch_market_price(symbol)).get("price", 0.0)
                await log_live_trade(
                    symbol,
                    side,
                    qty,
                    price,
                    self.config,
                    market="crypto",
                )
                return result
            except Exception as e:
                logging.error(f"place_order failed (attempt {attempt}): {e}")
                await asyncio.sleep(attempt)
        return {"status": "error"}

    async def cancel_order(self, order_id: str) -> Any:
        if self.simulation_mode:
            logging.info(f"CoinbaseClient SIM cancel {order_id}")
            return {"id": order_id, "status": "canceled"}
        for attempt in range(1, 4):
            try:
                return await self._run(self.client.cancel_orders, [order_id])
            except Exception as e:
                logging.error(f"cancel_order failed (attempt {attempt}): {e}")
                await asyncio.sleep(attempt)
        return {"status": "error"}

    async def fetch_market_price(self, product_id: str) -> Dict[str, float]:
        if self.simulation_mode:
            return {"price": 0.0, "bid": 0.0, "ask": 0.0}
        for attempt in range(1, 4):
            try:
                data = await self._run(self.client.get_market_trades, product_id, 1)
                bid = float(getattr(data, "best_bid", 0) or 0)
                ask = float(getattr(data, "best_ask", 0) or 0)
                price = (bid + ask) / 2 if bid and ask else bid or ask
                return {"price": price, "bid": bid, "ask": ask}
            except Exception as e:
                logging.error(f"fetch_market_price failed (attempt {attempt}): {e}")
                await asyncio.sleep(attempt)
        return {"price": 0.0, "bid": 0.0, "ask": 0.0}

    async def close(self) -> None:
        """Close the underlying HTTP session if available."""
        if self.client and hasattr(self.client, "session"):
            try:
                await asyncio.to_thread(self.client.session.close)
            except Exception as e:  # pragma: no cover - rarely triggered
                logging.error(f"Error closing Coinbase client session: {e}")
