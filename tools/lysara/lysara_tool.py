"""
tools/lysara/lysara_tool.py

Sylana Vessel tool handler for the Lysara Investments Tier 2 interface.

Register in Brain.create_default():
    from tools.lysara.lysara_tool import lysara_handler
    registry.register("lysara", lysara_handler)

Supported actions (via ToolRequest.action):
    health_check        — ping the daemon
    get_status          — full system snapshot (paused, regime, equity, markets)
    get_performance     — trade stats, P&L, win rate per market
    get_recent_trades   — last N trades  (parameters: limit=int)
    pause_all           — halt all trading  (parameters: reason=str)
    resume_all          — re-enable all trading
    pause_market        — pause one market  (parameters: market=str)
    resume_market       — resume one market (parameters: market=str)
    adjust_risk         — change live risk params
                          (parameters: market, risk_per_trade, max_daily_loss)
"""
from __future__ import annotations

from tools.tool_contract import ToolRequest, ToolResult
from tools.lysara.lysara_client import LysaraClient

_client = LysaraClient()


def _offline_error(data: dict) -> ToolResult:
    return ToolResult.error(
        "Lysara trading engine appears to be offline. Start it with `python main.py --simulate` "
        "or `python main.py --live` from the Lysara_Investments directory.",
        data,
    )


def lysara_handler(req: ToolRequest) -> ToolResult:
    try:
        action = req.action

        # ---- health_check ------------------------------------------------
        if action == "health_check":
            data = _client.health()
            if data.get("daemon_offline"):
                return _offline_error(data)
            paused = data.get("paused", False)
            mode = "simulation" if data.get("simulation_mode") else "LIVE"
            return ToolResult.success(
                data,
                f"Lysara daemon online — {'PAUSED' if paused else 'running'} in {mode} mode.",
            )

        # ---- get_status --------------------------------------------------
        if action == "get_status":
            data = _client.status()
            if data.get("daemon_offline"):
                return _offline_error(data)
            paused = data.get("paused", False)
            mode = "simulation" if data.get("simulation_mode") else "LIVE"
            markets = ", ".join(data.get("active_markets", [])) or "none"
            equity = data.get("equity", {})
            equity_str = "  ".join(f"{m}=${v:,.2f}" for m, v in equity.items()) or "unavailable"
            regime = data.get("regime", "unknown")
            uptime = data.get("uptime_seconds", 0)
            return ToolResult.success(
                data,
                f"Trading engine {'PAUSED' if paused else 'RUNNING'} in {mode} mode. "
                f"Markets: {markets}. Equity: {equity_str}. Regime: {regime}. "
                f"Uptime: {uptime // 3600}h {(uptime % 3600) // 60}m.",
            )

        # ---- get_performance ---------------------------------------------
        if action == "get_performance":
            data = _client.performance()
            if data.get("daemon_offline"):
                return _offline_error(data)
            rows = data.get("by_market", [])
            if not rows:
                return ToolResult.success(data, "No completed trades recorded yet.")
            parts = [
                f"{r['market']}/{r['side']}: {r['total_trades']} trades | "
                f"W/L={r['wins']}/{r['losses']} ({r.get('win_rate_pct', 0)}%) | "
                f"PnL=${r['total_pnl']:+.2f}"
                for r in rows
            ]
            return ToolResult.success(data, " || ".join(parts))

        # ---- get_recent_trades -------------------------------------------
        if action == "get_recent_trades":
            limit = int(req.parameters.get("limit", 10))
            data = _client.recent_trades(limit)
            if data.get("daemon_offline"):
                return _offline_error(data)
            count = data.get("count", 0)
            return ToolResult.success(data, f"{count} most recent trades retrieved.")

        # ---- pause_all ---------------------------------------------------
        if action == "pause_all":
            reason = str(req.parameters.get("reason", "Sylana directive"))
            data = _client.pause(reason)
            if data.get("daemon_offline"):
                return _offline_error(data)
            return ToolResult.success(data, f"All trading halted. Reason: {reason}")

        # ---- resume_all --------------------------------------------------
        if action == "resume_all":
            data = _client.resume()
            if data.get("daemon_offline"):
                return _offline_error(data)
            return ToolResult.success(data, "Trading resumed across all markets.")

        # ---- pause_market ------------------------------------------------
        if action == "pause_market":
            market = str(req.parameters.get("market", "crypto"))
            data = _client.pause_market(market)
            if data.get("daemon_offline"):
                return _offline_error(data)
            return ToolResult.success(data, f"{market.capitalize()} strategies paused.")

        # ---- resume_market -----------------------------------------------
        if action == "resume_market":
            market = str(req.parameters.get("market", "crypto"))
            data = _client.resume_market(market)
            if data.get("daemon_offline"):
                return _offline_error(data)
            return ToolResult.success(data, f"{market.capitalize()} strategies resumed.")

        # ---- adjust_risk -------------------------------------------------
        if action == "adjust_risk":
            market = str(req.parameters.get("market", "crypto"))
            risk_per_trade = req.parameters.get("risk_per_trade")
            max_daily_loss = req.parameters.get("max_daily_loss")
            if risk_per_trade is None and max_daily_loss is None:
                return ToolResult.error(
                    "adjust_risk requires at least one of: risk_per_trade, max_daily_loss"
                )
            data = _client.adjust_risk(
                market=market,
                risk_per_trade=float(risk_per_trade) if risk_per_trade is not None else None,
                max_daily_loss=float(max_daily_loss) if max_daily_loss is not None else None,
            )
            if data.get("daemon_offline"):
                return _offline_error(data)
            n = data.get("risk_managers_updated", 0)
            changes = []
            if risk_per_trade is not None:
                changes.append(f"risk_per_trade={float(risk_per_trade):.3f}")
            if max_daily_loss is not None:
                changes.append(f"max_daily_loss={float(max_daily_loss):.2f}")
            return ToolResult.success(
                data,
                f"Risk updated for {market} ({n} manager(s)): {', '.join(changes)}.",
            )

        return ToolResult.error(f"Unknown lysara action: '{action}'. "
                                f"Valid actions: health_check, get_status, get_performance, "
                                f"get_recent_trades, pause_all, resume_all, pause_market, "
                                f"resume_market, adjust_risk.")

    except Exception as e:
        return ToolResult.error(f"Lysara tool error: {e}")
