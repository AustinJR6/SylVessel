# api/base_api.py

import aiohttp
import asyncio
import logging
from urllib.parse import urljoin

class BaseAPI:
    """
    Shared HTTP client with retry logic and session management.
    Exchanges should subclass this and provide auth headers as needed.
    """

    def __init__(self, base_url: str, session: aiohttp.ClientSession = None):
        self.base_url = base_url
        self.session = session or aiohttp.ClientSession()

    async def get(self, path: str, headers: dict = None) -> dict:
        return await self._request('GET', path, headers=headers)

    async def post(self, path: str, body: dict = None, headers: dict = None) -> dict:
        return await self._request('POST', path, body=body, headers=headers)

    async def _request(
        self,
        method: str,
        path: str,
        body: dict = None,
        headers: dict = None
    ) -> dict:
        url = urljoin(self.base_url, path)
        for attempt in range(1, 4):
            try:
                if method == 'GET':
                    resp = await self.session.get(url, headers=headers)
                else:
                    resp = await self.session.post(url, json=body or {}, headers=headers)
                resp.raise_for_status()
                return await resp.json()
            except Exception as e:
                logging.error(f"{method} {url} failed (attempt {attempt}): {e}")
                await asyncio.sleep(attempt)  # backoff: 1s, 2s, 3s
        logging.error(f"{method} {url} failed after 3 attempts; returning empty dict.")
        return {}

    async def close(self):
        """Close underlying HTTP session."""
        await self.session.close()
