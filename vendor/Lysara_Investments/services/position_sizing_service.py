from __future__ import annotations

from typing import Any

import numpy as np


class PositionSizingService:
    def __init__(self, state, db=None, exposure_service=None):
        self.state = state
        self.db = db
        self.exposure_service = exposure_service

    def _closed_trade_rows(self, market: str, symbol: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        max_rows = max(5, min(int(limit or 50), 200))
        params: list[Any] = [market]
        query = """
            SELECT symbol, side, quantity, price, profit_loss, timestamp
            FROM trades
            WHERE market = ? AND profit_loss IS NOT NULL
        """
        if symbol:
            query += " AND symbol = ?"
            params.append(str(symbol).upper())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max_rows)
        return self.db.fetch_all(query, tuple(params))

    def _kelly_profile(self, market: str, symbol: str | None = None) -> dict[str, Any]:
        rows = self._closed_trade_rows(
            market=market,
            symbol=symbol,
            limit=int(self.state.config.get("position_sizing_min_history_trades", 8)) * 4,
        )
        min_history = max(1, int(self.state.config.get("position_sizing_min_history_trades", 8)))
        max_fraction = max(0.01, float(self.state.config.get("position_sizing_max_kelly_fraction", 0.25)))

        if len(rows) < min_history:
            return {
                "active": False,
                "fraction": 0.0,
                "scale": 1.0,
                "trade_count": len(rows),
                "reason": "insufficient_history",
            }

        pnls = [float(row.get("profit_loss") or 0.0) for row in rows]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [abs(pnl) for pnl in pnls if pnl < 0]
        if not wins or not losses:
            return {
                "active": True,
                "fraction": 0.0,
                "scale": 0.5,
                "trade_count": len(rows),
                "reason": "unbalanced_trade_history",
            }

        win_rate = len(wins) / len(pnls)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        raw_fraction = win_rate - ((1.0 - win_rate) / payoff_ratio) if payoff_ratio > 0 else 0.0
        capped_fraction = max(0.0, min(raw_fraction, max_fraction))
        normalized = (capped_fraction / max_fraction) if max_fraction > 0 else 0.0
        return {
            "active": True,
            "fraction": round(capped_fraction, 4),
            "scale": round(max(0.35, normalized), 4),
            "trade_count": len(rows),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "payoff_ratio": round(payoff_ratio, 4),
            "reason": "history_based",
        }

    def _relative_volatility(self, price: float, price_history: list[float], period: int = 14) -> float:
        if price <= 0 or len(price_history) < 2:
            return 0.0
        diffs = np.diff(price_history[-max(period, 2):])
        if len(diffs) == 0:
            return 0.0
        return float(np.std(diffs) / price)

    def _drawdown_scale(self, risk_manager: Any) -> float:
        max_daily_loss = abs(float(getattr(risk_manager, "max_daily_loss", 0.0) or 0.0))
        daily_loss = float(getattr(risk_manager, "daily_loss", 0.0) or 0.0)
        if max_daily_loss <= 0 or daily_loss >= 0:
            return 1.0
        ratio = min(abs(daily_loss) / max_daily_loss, 1.0)
        return round(max(0.25, 1.0 - (0.75 * ratio)), 4)

    def _heat_scale(self, exposure: dict[str, Any]) -> float:
        total_heat = float(exposure.get("total_effective_heat_pct") or 0.0)
        if total_heat <= 0.6:
            return 1.0
        return round(max(0.35, 1.0 - min(total_heat - 0.6, 0.65)), 4)

    def _exposure_snapshot(self, market: str) -> dict[str, Any]:
        if self.exposure_service is not None:
            return self.exposure_service.get_exposure(market=market)
        return {
            "portfolio_value": 0.0,
            "cash": 0.0,
            "holdings_value": 0.0,
            "gross_exposure_pct": 0.0,
            "heat_score": 0.0,
            "total_effective_heat_pct": 0.0,
            "max_single_position_pct": 0.0,
            "max_total_gross_exposure_pct": 0.0,
            "positions": [],
        }

    def compute_order_size(
        self,
        *,
        market: str,
        symbol: str,
        side: str,
        price: float,
        confidence: float,
        risk_manager: Any,
        price_history: list[float] | None = None,
        desired_qty: float | None = None,
        current_position_qty: float = 0.0,
    ) -> dict[str, Any]:
        normalized_market = str(market or "crypto").strip().lower()
        normalized_symbol = str(symbol or "").strip().upper()
        normalized_side = str(side or "buy").strip().lower()
        history = list(price_history or [])
        exposure = self._exposure_snapshot(normalized_market)

        if price <= 0:
            return {"quantity": 0.0, "reason": "invalid_price", "exposure": exposure}

        base_qty = max(float(getattr(risk_manager, "get_position_size")(price) or 0.0), 0.0)
        requested_qty = float(desired_qty or 0.0) if desired_qty is not None else None
        confidence_scale = round(max(min(float(confidence or 0.0), 1.0), 0.25), 4)
        relative_vol = self._relative_volatility(price, history)
        target_relative_vol = 0.02
        if relative_vol <= 0 or relative_vol <= target_relative_vol:
            volatility_scale = 1.0
        else:
            volatility_scale = round(max(target_relative_vol / relative_vol, 0.15), 4)
        drawdown_scale = self._drawdown_scale(risk_manager)
        heat_scale = self._heat_scale(exposure)
        kelly = self._kelly_profile(normalized_market, normalized_symbol)
        kelly_scale = float(kelly.get("scale") or 1.0)

        current_position_value = 0.0
        for row in exposure.get("positions") or []:
            if str((row or {}).get("symbol") or "").upper() == normalized_symbol:
                current_position_value = float((row or {}).get("market_value") or 0.0)
                break

        portfolio_value = float(exposure.get("portfolio_value") or 0.0)
        cash = float(exposure.get("cash") or 0.0)
        gross_exposure_pct = float(exposure.get("gross_exposure_pct") or 0.0)
        max_single_position_pct = float(exposure.get("max_single_position_pct") or 0.0)
        max_total_gross_exposure_pct = float(exposure.get("max_total_gross_exposure_pct") or 0.0)

        capped_by: list[str] = []
        if normalized_side == "sell":
            raw_qty = requested_qty if requested_qty is not None else float(current_position_qty or 0.0)
            quantity = min(max(raw_qty, 0.0), max(float(current_position_qty or 0.0), 0.0))
            if quantity <= 0:
                return {
                    "quantity": 0.0,
                    "reason": "no_open_position",
                    "exposure": exposure,
                    "kelly": kelly,
                }
        else:
            raw_qty = requested_qty if requested_qty is not None else base_qty * confidence_scale * volatility_scale * drawdown_scale * heat_scale * kelly_scale
            max_position_value = (portfolio_value * max_single_position_pct) if portfolio_value > 0 and max_single_position_pct > 0 else cash
            remaining_single_value = max(max_position_value - current_position_value, 0.0)
            max_gross_value = (portfolio_value * max_total_gross_exposure_pct) if portfolio_value > 0 and max_total_gross_exposure_pct > 0 else cash
            remaining_gross_value = max(max_gross_value - (gross_exposure_pct * portfolio_value), 0.0) if portfolio_value > 0 else cash
            capital_cap_value = min(
                [value for value in (cash, remaining_single_value, remaining_gross_value) if value >= 0.0] or [0.0]
            )
            quantity = min(max(raw_qty, 0.0), (capital_cap_value / price) if price > 0 else 0.0)
            if quantity < max(raw_qty, 0.0):
                if capital_cap_value <= cash + 1e-9:
                    capped_by.append("cash")
                if remaining_single_value <= capital_cap_value + 1e-9:
                    capped_by.append("single_position_cap")
                if remaining_gross_value <= capital_cap_value + 1e-9:
                    capped_by.append("gross_exposure_cap")

        projected_position_value = current_position_value
        projected_gross_exposure_pct = gross_exposure_pct
        if normalized_side == "buy":
            projected_position_value = current_position_value + (quantity * price)
            projected_gross_exposure_pct = ((gross_exposure_pct * portfolio_value) + (quantity * price)) / portfolio_value if portfolio_value > 0 else 0.0

        return {
            "market": normalized_market,
            "symbol": normalized_symbol,
            "side": normalized_side,
            "quantity": round(max(quantity, 0.0), 8),
            "notional": round(max(quantity, 0.0) * price, 4),
            "base_quantity": round(base_qty, 8),
            "raw_quantity": round(max(raw_qty, 0.0), 8),
            "scales": {
                "confidence": confidence_scale,
                "volatility": volatility_scale,
                "drawdown": drawdown_scale,
                "heat": heat_scale,
                "kelly": kelly_scale,
            },
            "kelly": kelly,
            "relative_volatility": round(relative_vol, 6),
            "projected_position_pct": round((projected_position_value / portfolio_value), 4) if portfolio_value > 0 else 0.0,
            "projected_gross_exposure_pct": round(projected_gross_exposure_pct, 4),
            "capped_by": capped_by,
            "exposure": exposure,
        }
