"""
tools/lysara/lysara_client.py

Synchronous HTTP client for the Lysara Tier 1 Control API.

Uses only stdlib (urllib) so no extra dependency is needed in Sylana_Vessel.
brain.think() is synchronous, so this client is synchronous too.

Configuration (via environment variables or .env):
    LYSARA_CONTROL_URL     Base URL of the control API  (default: http://127.0.0.1:18791)
    LYSARA_CONTROL_SECRET  Shared secret for X-API-Key header (default: empty = open)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class LysaraClient:
    def __init__(self):
        self.base_url = os.getenv("LYSARA_CONTROL_URL", "http://127.0.0.1:18791").rstrip("/")
        self.api_key = os.getenv("LYSARA_CONTROL_SECRET", "")
        self.timeout = int(os.getenv("LYSARA_CONTROL_TIMEOUT", "5"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            return {"error": f"HTTP {e.code}: {body_text}", "daemon_offline": e.code >= 500}
        except OSError:
            # Connection refused / timeout — daemon isn't running
            return {"error": "Connection refused — Lysara daemon may be offline.", "daemon_offline": True}
        except Exception as e:
            return {"error": str(e), "daemon_offline": True}

    # ------------------------------------------------------------------
    # Read endpoints
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def status(self) -> Dict[str, Any]:
        return self._request("GET", "/status")

    def performance(self) -> Dict[str, Any]:
        return self._request("GET", "/performance")

    def recent_trades(self, limit: int = 10) -> Dict[str, Any]:
        return self._request("GET", f"/trades/recent?limit={limit}")

    # ------------------------------------------------------------------
    # Command endpoints
    # ------------------------------------------------------------------

    def pause(self, reason: str = "Sylana directive") -> Dict[str, Any]:
        return self._request("POST", "/pause", {"reason": reason})

    def resume(self) -> Dict[str, Any]:
        return self._request("POST", "/resume")

    def pause_market(self, market: str) -> Dict[str, Any]:
        return self._request("POST", "/strategy/pause", {"market": market})

    def resume_market(self, market: str) -> Dict[str, Any]:
        return self._request("POST", "/strategy/resume", {"market": market})

    def adjust_risk(
        self,
        market: str,
        risk_per_trade: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"market": market}
        if risk_per_trade is not None:
            body["risk_per_trade"] = risk_per_trade
        if max_daily_loss is not None:
            body["max_daily_loss"] = max_daily_loss
        return self._request("POST", "/risk/adjust", body)
