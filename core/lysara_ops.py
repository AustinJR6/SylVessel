from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen


class LysaraOpsError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = int(status_code)
        self.message = message
        super().__init__(message)


@dataclass
class LysaraOpsClient:
    base_url: str
    api_key: str = ""
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> Optional["LysaraOpsClient"]:
        base_url = (os.getenv("LYSARA_OPS_BASE_URL") or "").strip().rstrip("/")
        if not base_url:
            return None
        return cls(
            base_url=base_url,
            api_key=(os.getenv("LYSARA_OPS_API_KEY") or "").strip(),
            timeout_seconds=max(5, min(int(os.getenv("LYSARA_OPS_TIMEOUT_SECONDS", "20")), 60)),
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        expected: Optional[set[int]] = None,
    ) -> Dict[str, Any]:
        expected_codes = expected or {200}
        url = f"{self.base_url}{path}"
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urlencode(clean_params, doseq=True)}"

        headers = {"Accept": "application/json"}
        data = None
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = UrlRequest(url=url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                status = int(resp.status)
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            raise LysaraOpsError(exc.code, detail[:500] or f"HTTP {exc.code}") from exc
        except URLError as exc:
            raise LysaraOpsError(503, f"Lysara ops unavailable: {exc.reason}") from exc
        except Exception as exc:
            raise LysaraOpsError(500, f"Lysara ops request failed: {exc}") from exc

        if status not in expected_codes:
            raise LysaraOpsError(status, f"Unexpected Lysara status {status}")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception as exc:
            raise LysaraOpsError(502, f"Lysara ops returned non-JSON payload: {raw[:200]}") from exc

    def get_health(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/health", expected={200})

    def get_status(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/status", expected={200})

    def get_portfolio(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/portfolio", expected={200})

    def get_positions(self, market: Optional[str] = None) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/positions", params={"market": market}, expected={200})

    def get_recent_trades(self, limit: int = 20, market: Optional[str] = None) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/trades/recent", params={"limit": limit, "market": market}, expected={200})

    def get_market_snapshot(self, symbols: Optional[str] = None) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/market-snapshot", params={"symbols": symbols}, expected={200})

    def get_incidents(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/incidents", params={"status": status, "limit": limit}, expected={200})

    def get_research(self, market: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/research", params={"market": market, "limit": limit}, expected={200})

    def get_journal(self, limit: int = 50) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/ops/journal", params={"limit": limit}, expected={200})

    def pause_trading(self, reason: str, market: str = "all", actor: str = "operator") -> Dict[str, Any]:
        return self._request("POST", "/api/v1/ops/pause", payload={"reason": reason, "market": market, "actor": actor}, expected={200})

    def resume_trading(self, market: str = "all", actor: str = "operator") -> Dict[str, Any]:
        return self._request("POST", "/api/v1/ops/resume", payload={"market": market, "actor": actor}, expected={200})

    def adjust_risk(
        self,
        *,
        market: str,
        actor: str = "operator",
        risk_per_trade: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/ops/risk",
            payload={
                "market": market,
                "actor": actor,
                "risk_per_trade": risk_per_trade,
                "max_daily_loss": max_daily_loss,
            },
            expected={200},
        )

    def update_strategy_params(
        self,
        *,
        market: str,
        actor: str = "operator",
        strategy_name: Optional[str] = None,
        enabled: Optional[bool] = None,
        symbol_controls: Optional[Dict[str, bool]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/ops/strategy",
            payload={
                "market": market,
                "actor": actor,
                "strategy_name": strategy_name,
                "enabled": enabled,
                "symbol_controls": symbol_controls or {},
                "params": params or {},
            },
            expected={200},
        )

    def submit_trade_intent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/ops/trade-intents", payload=payload, expected={200})

    def record_research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/ops/research", payload=payload, expected={200})

    def record_journal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/ops/journal", payload=payload, expected={200})

    def acknowledge_incident(self, incident_id: int, actor: str = "operator") -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/ops/incidents/{incident_id}/ack", payload={"actor": actor}, expected={200})

    def resolve_incident(self, incident_id: int, actor: str = "operator") -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/ops/incidents/{incident_id}/resolve", payload={"actor": actor}, expected={200})
