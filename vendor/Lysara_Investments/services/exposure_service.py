from __future__ import annotations

from typing import Any

from data.price_cache import get_price


def _asset_group(symbol: str) -> str:
    asset = str(symbol or "").split("-", 1)[0].upper()
    if asset in {"BTC", "ETH"}:
        return "majors"
    return "alts"


def _correlation_proxy(symbol_a: str, symbol_b: str) -> float:
    if symbol_a == symbol_b:
        return 1.0
    group_a = _asset_group(symbol_a)
    group_b = _asset_group(symbol_b)
    if group_a == group_b == "majors":
        return 0.55
    if group_a == group_b == "alts":
        return 0.8
    return 0.65


class ExposureService:
    def __init__(self, state):
        self.state = state

    def _portfolio_snapshot(self) -> dict[str, Any]:
        portfolio = self.state.get_sim_portfolio()
        if portfolio is None:
            fallback_value = float(
                self.state.equity.get("crypto")
                or next(iter(self.state.equity.values()), 0.0)
                or self.state.config.get("starting_balance", 0.0)
                or 0.0
            )
            return {"portfolio_value": fallback_value, "cash": fallback_value, "buying_power": fallback_value}
        return portfolio.account_snapshot()

    def get_exposure(self, market: str = "crypto") -> dict[str, Any]:
        normalized_market = str(market or "crypto").strip().lower()
        snapshot = self._portfolio_snapshot()
        portfolio_value = float(snapshot.get("portfolio_value") or 0.0)
        cash = float(snapshot.get("cash") or snapshot.get("buying_power") or 0.0)
        positions: list[dict[str, Any]] = []

        if normalized_market == "crypto":
            portfolio = self.state.get_sim_portfolio()
            holdings = portfolio.holdings_snapshot() if portfolio else {}
            for symbol, quantity in holdings.items():
                price_entry = get_price(symbol)
                price = float((price_entry or {}).get("price") or 0.0)
                market_value = float(quantity) * price if price > 0 else 0.0
                positions.append(
                    {
                        "symbol": str(symbol).upper(),
                        "quantity": float(quantity),
                        "price": price,
                        "market_value": market_value,
                    }
                )

        gross_value = sum(max(float(item["market_value"]), 0.0) for item in positions)
        gross_exposure_pct = (gross_value / portfolio_value) if portfolio_value > 0 else 0.0

        for item in positions:
            market_value = float(item["market_value"])
            weight_pct = (market_value / portfolio_value) if portfolio_value > 0 else 0.0
            symbol = str(item["symbol"])
            correlated = sum(
                _correlation_proxy(symbol, str(other["symbol"])) * (float(other["market_value"]) / portfolio_value)
                for other in positions
                if other is not item and portfolio_value > 0
            )
            effective_weight_pct = weight_pct + correlated
            item["weight_pct"] = round(weight_pct, 4)
            item["effective_weight_pct"] = round(effective_weight_pct, 4)

        positions.sort(key=lambda item: float(item.get("market_value") or 0.0), reverse=True)
        heat_score = max((float(item.get("effective_weight_pct") or 0.0) for item in positions), default=0.0)
        total_effective_heat_pct = min(sum(float(item.get("effective_weight_pct") or 0.0) for item in positions), 2.0)
        max_single_position_pct = float(self.state.config.get("ops_max_single_position_pct", 20.0)) / 100.0
        max_total_gross_exposure_pct = float(self.state.config.get("ops_max_total_gross_exposure_pct", 100.0)) / 100.0

        return {
            "market": normalized_market,
            "portfolio_value": round(portfolio_value, 4),
            "cash": round(cash, 4),
            "holdings_value": round(gross_value, 4),
            "gross_exposure_pct": round(gross_exposure_pct, 4),
            "heat_score": round(heat_score, 4),
            "total_effective_heat_pct": round(total_effective_heat_pct, 4),
            "max_single_position_pct": round(max_single_position_pct, 4),
            "max_total_gross_exposure_pct": round(max_total_gross_exposure_pct, 4),
            "positions": positions,
        }
