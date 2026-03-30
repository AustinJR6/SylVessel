import asyncio
import logging
from typing import Dict
import aiohttp

COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

class MarketStateMonitor:
    """Monitors overall crypto market state using CoinGecko data."""

    def __init__(self):
        self.state = {}
        self._running = True

    async def fetch_state(self) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(COINGECKO_GLOBAL_URL) as resp:
                data = await resp.json()
                return data.get("data", {})

    async def run(self, interval: int = 300):
        while self._running:
            try:
                self.state = await self.fetch_state()
                btc_dom = self.state.get("market_cap_percentage", {}).get("btc", 0)
                total_cap = self.state.get("total_market_cap", {}).get("usd", 0)
                logging.info(
                    f"MarketState BTC dom={btc_dom:.2f}% total_cap={total_cap:.0f}"
                )
            except Exception as e:
                logging.error(f"Market state fetch failed: {e}")
            await asyncio.sleep(interval)

    def stop(self):
        self._running = False
