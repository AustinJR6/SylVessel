"""Async Robinhood crypto client.

This client wraps Robinhood's crypto trading API behind the same interface the
existing crypto strategies already expect: account info, holdings, market
price lookup, and order placement.

The endpoint paths and Ed25519 request-signing shape are based on Robinhood's
published crypto API surface and public SDK implementations.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import aiohttp
from nacl.signing import SigningKey

from api.base_api import BaseAPI
from api.binance_public import fetch_binance_public_price
from data.price_cache import get_price
from utils.guardrails import log_live_trade


class RobinhoodCryptoClient(BaseAPI):
    """Simplified async client for Robinhood crypto trading."""

    def __init__(
        self,
        api_key: str = "",
        private_key: str = "",
        *,
        base_url: str = "https://trading.robinhood.com",
        simulation_mode: bool = True,
        portfolio=None,
        config: Optional[Dict[str, Any]] = None,
        trade_cooldown: int = 30,
    ) -> None:
        super().__init__(base_url.rstrip("/") + "/")
        self.api_key = api_key
        self.private_key = private_key
        self.simulation_mode = simulation_mode
        self.portfolio = portfolio
        self.config = config or {}
        self.trade_cooldown = trade_cooldown
        self._last_trade: Dict[str, float] = {}
        self._mock_equity = 10000.0
        self._mock_holdings: Dict[str, float] = {}
        self._signing_key: SigningKey | None = None
        if not simulation_mode and api_key and private_key:
            self._signing_key = self._decode_signing_key(private_key)

    @staticmethod
    def _decode_signing_key(key_material: str) -> SigningKey:
        seed = base64.b64decode(key_material)
        if len(seed) != 32:
            raise ValueError("Robinhood signing key must decode to 32 bytes")
        return SigningKey(seed)

    @staticmethod
    def _coerce_decimal(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _asset_code(symbol: str) -> str:
        return symbol.split("-", 1)[0].upper()

    @staticmethod
    def _symbol(symbol: str) -> str:
        sym = symbol.upper()
        return sym if "-" in sym else f"{sym}-USD"

    def _headers(self, path_with_query: str, body: str, method: str) -> Dict[str, str]:
        if not self._signing_key:
            raise RuntimeError("Robinhood crypto client is missing signing credentials")
        timestamp = int(time.time())
        message = f"{self.api_key}{timestamp}{path_with_query}{method.upper()}{body}"
        signed = self._signing_key.sign(message.encode("utf-8")).signature
        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed).decode("utf-8"),
            "x-timestamp": str(timestamp),
            "Content-Type": "application/json",
        }

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        clean_params = {k: v for k, v in params.items() if v is not None}
        if clean_params:
            encoded = urlencode(clean_params, doseq=True)
            path_with_query = f"{path}?{encoded}"
        else:
            path_with_query = path
        payload = json.dumps(body or {}, separators=(",", ":")) if body is not None else ""
        headers = self._headers(path_with_query, payload, method)
        url = f"{self.base_url.rstrip('/')}/{path_with_query.lstrip('/')}"

        backoff = 1
        for attempt in range(1, 4):
            try:
                async with self.session.request(
                    method.upper(),
                    url,
                    headers=headers,
                    json=body if body is not None else None,
                ) as resp:
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        text = await resp.text()
                        data = {"raw_text": text}
                    if resp.status >= 500:
                        logging.warning(
                            "Robinhood server error %s on %s %s (attempt %s)",
                            resp.status,
                            method.upper(),
                            path,
                            attempt,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8)
                        continue
                    if resp.status >= 400:
                        logging.error(
                            "Robinhood HTTP %s on %s %s: %s",
                            resp.status,
                            method.upper(),
                            path,
                            data,
                        )
                        return {"error": resp.status, "message": data}
                    return data if isinstance(data, dict) else {"results": data}
            except aiohttp.ClientError as exc:
                logging.warning(
                    "Robinhood request failed on %s %s (attempt %s): %s",
                    method.upper(),
                    path,
                    attempt,
                    exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8)
        return {"error": "max_retries", "message": f"{method.upper()} {path} failed"}

    async def fetch_account_info(self) -> Dict[str, Any]:
        if self.simulation_mode:
            logging.debug("RobinhoodCryptoClient: simulation mode returning mock account info")
            if self.portfolio:
                return self.portfolio.account_snapshot()
            return {"currency": "USD", "balance": self._mock_equity, "buying_power": self._mock_equity}
        data = await self._request_json("GET", "/api/v1/crypto/trading/accounts/")
        buying_power = self._coerce_decimal(data.get("buying_power"))
        return {
            **data,
            "balance": buying_power,
            "buying_power": buying_power,
            "currency": data.get("buying_power_currency", "USD"),
        }

    async def get_holdings(self) -> Dict[str, float]:
        if self.simulation_mode:
            logging.debug("RobinhoodCryptoClient: simulation mode returning mock holdings")
            if self.portfolio:
                return self.portfolio.holdings_snapshot()
            return self._mock_holdings
        data = await self._request_json("GET", "/api/v1/crypto/trading/holdings/")
        holdings: Dict[str, float] = {}
        for item in data.get("results", []):
            asset_code = item.get("asset_code")
            qty = self._coerce_decimal(item.get("quantity_available_for_trading"))
            if asset_code and qty:
                holdings[str(asset_code).upper()] = qty
        return holdings

    async def fetch_market_price(self, symbol: str) -> Dict[str, float]:
        cached = get_price(symbol)
        if self.simulation_mode:
            if cached:
                price = self._coerce_decimal(cached.get("price"))
                if price > 0:
                    return {"price": price, "bid": price, "ask": price, "source": str(cached.get("source") or "cache")}
            data = await fetch_binance_public_price(symbol)
            price = self._coerce_decimal(data.get("price"))
            if price > 0:
                return {
                    "price": price,
                    "bid": self._coerce_decimal(data.get("bid")),
                    "ask": self._coerce_decimal(data.get("ask")),
                    "source": str(data.get("source") or "binance_public"),
                }
            return {"price": 0.0, "bid": 0.0, "ask": 0.0, "source": "binance_public"}

        canonical = self._symbol(symbol)
        data = await self._request_json(
            "GET",
            "/api/v1/crypto/marketdata/best_bid_ask/",
            params={"symbol": canonical},
        )
        result = {}
        if data.get("results"):
            result = data["results"][0] or {}
        bid = self._coerce_decimal(result.get("bid_inclusive_of_sell_spread"))
        ask = self._coerce_decimal(result.get("ask_inclusive_of_buy_spread"))
        price = self._coerce_decimal(result.get("price"))
        if not price:
            price = (bid + ask) / 2 if bid and ask else bid or ask
        return {"price": price, "bid": bid, "ask": ask}

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        **kwargs,
    ) -> Any:
        side_lower = str(side).lower()
        if self.simulation_mode:
            logging.info("RobinhoodCryptoClient SIM %s %s %s", side_lower, qty, symbol)
            price = 0.0
            if self.portfolio:
                price = (await self.fetch_market_price(symbol)).get("price", 0.0)
                execution = self.portfolio.execute_trade(symbol, side_lower, qty, price, kwargs.get("confidence", 0.0))
                if execution.get("status") != "filled":
                    return {"id": "sim_order", "state": "rejected", **execution}
            return {"id": "sim_order", "state": "filled", "filled_asset_quantity": qty}

        now = asyncio.get_event_loop().time()
        last = self._last_trade.get(symbol)
        if last and now - last < self.trade_cooldown:
            logging.warning("Duplicate Robinhood trade blocked for %s", symbol)
            return {"status": "blocked", "reason": "duplicate"}
        self._last_trade[symbol] = now

        canonical = self._symbol(symbol)
        body = {
            "symbol": canonical,
            "client_order_id": kwargs.get("client_order_id") or str(uuid.uuid4()),
            "side": side_lower,
            "type": order_type.lower(),
            "market_order_config": {
                "asset_quantity": str(qty),
            },
        }
        result = await self._request_json("POST", "/api/v1/crypto/trading/orders/", body=body)
        if result.get("error"):
            return result
        price = (await self.fetch_market_price(canonical)).get("price", 0.0)
        await log_live_trade(
            canonical,
            side_lower,
            qty,
            price,
            self.config,
            market="crypto",
            confidence=kwargs.get("confidence"),
        )
        return result

    async def cancel_order(self, order_id: str) -> Any:
        if self.simulation_mode:
            return {"id": order_id, "status": "canceled"}
        return await self._request_json(
            "POST",
            f"/api/v1/crypto/trading/orders/{order_id}/cancel/",
        )

    async def fetch_holdings(self) -> Dict[str, float]:
        return await self.get_holdings()
