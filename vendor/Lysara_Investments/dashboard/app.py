"""Main entry for the Streamlit dashboard with real-time updates."""

from __future__ import annotations

import datetime
import os
import sys

import streamlit as st

# Ensure project root is importable when running this file directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.config_manager import ConfigManager
from dashboard.controls.risk_controls import show_risk_controls
from dashboard.controls.trading_controls import show_trading_controls
from dashboard.utils import (
    PortfolioManager,
    auto_refresh,
    get_ai_thoughts,
    get_equity,
    get_equity_curve,
    get_last_agent_decision,
    get_last_trade,
    get_last_trade_per_market,
    get_log_lines,
    get_performance_metrics,
    get_sentiment_data,
    get_trade_history,
    load_control_flags,
    mock_trade_history,
)
from dashboard.utils.live_market_data import get_live_chart_data, pick_symbol
from services.ai_strategist import get_last_decision
from dashboard.views import (
    show_agent_status,
    show_ai_thought_feed,
    show_conviction_heatmap,
    show_crypto_view,
    show_equity_curve,
    show_forex_view,
    show_log_view,
    show_performance_view,
    show_portfolio_table,
    show_sim_summary,
    show_stocks_view,
    show_trade_history,
)


def _chart_symbols(config: dict) -> tuple[str, str, str]:
    crypto_candidates = config.get("TRADE_SYMBOLS", []) or config.get("crypto_settings", {}).get("trade_symbols", [])
    stock_candidates = config.get("stocks_settings", {}).get("trade_symbols", [])
    forex_candidates = config.get("forex_settings", {}).get("trade_symbols", [])
    return (
        pick_symbol(crypto_candidates, crypto_candidates[0] if crypto_candidates else "BTC-USD"),
        pick_symbol(stock_candidates, stock_candidates[0] if stock_candidates else "AAPL"),
        pick_symbol(forex_candidates, forex_candidates[0] if forex_candidates else "EUR_USD"),
    )


def main():
    st.set_page_config(page_title="Lysara Dashboard", layout="wide")
    auto_refresh(10)
    st.title("Lysara Investments Dashboard")

    config = ConfigManager().load_config()
    pm = PortfolioManager(config)
    forex_enabled = config.get("FOREX_ENABLED", False)

    mode = "LIVE" if not config.get("simulation_mode", True) else "SIM"
    banner_color = "red" if mode == "LIVE" else "green"
    st.markdown(
        f"<div style='background-color:{banner_color};padding:6px;text-align:center;color:white;'>Trading Mode: {mode}</div>",
        unsafe_allow_html=True,
    )
    if mode == "SIM":
        st.sidebar.success("SIMULATION MODE ON")
    else:
        st.sidebar.error("LIVE MODE ACTIVE")

    timeframe = st.sidebar.selectbox("Chart Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2)
    crypto_symbol, stock_symbol, forex_symbol = _chart_symbols(config)
    st.sidebar.caption(f"Chart symbols: {crypto_symbol}, {stock_symbol}" + (f", {forex_symbol}" if forex_enabled else ""))

    if config.get("SHOW_MANUAL_TRADING_UI", False):
        show_trading_controls(pm.sim_portfolio)
        show_risk_controls()

    flags = load_control_flags()
    if flags:
        st.sidebar.markdown("### Active Flags")
        st.sidebar.json(flags)

    auto_mode = st.sidebar.checkbox("Autonomous Mode", value=True)

    with st.spinner("Loading data..."):
        try:
            last_trade = get_last_trade()
            trade_history = get_trade_history()
            metrics = get_performance_metrics()
            equity = get_equity()
            equity_curve_data = get_equity_curve()
            sentiment = get_sentiment_data()
            logs = get_log_lines()
            ai_feed = get_ai_thoughts()
            real_holdings = pm.get_account_holdings()
            sim_data = pm.get_simulated_portfolio() if config.get("simulation_mode", True) else None
        except Exception as exc:
            st.error(f"Data load failed: {exc}")
            last_trade, trade_history, metrics = None, [], {}
            equity, equity_curve_data, sentiment = 0.0, [], {}
            logs, ai_feed = [], []
            real_holdings = {"crypto": [], "stocks": [], "forex": []}
            sim_data = None

    last_updated = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if not trade_history:
        trade_history = mock_trade_history()

    top = st.columns(3)
    if last_trade:
        top[0].write(
            f"**Last Trade:** {last_trade['timestamp']} {last_trade['symbol']} {last_trade['side']} "
            f"{last_trade['quantity']} @ {last_trade['price']} PnL={last_trade['pnl']}"
        )
    else:
        top[0].write("**Last Trade:** None")
    top[1].metric("Portfolio Equity", equity)
    top[2].metric("Open Risk", metrics.get("open_risk", 0.0))

    decision = get_last_decision()
    st.markdown("### AI Strategist Last Decision")
    if decision:
        conf = decision["decision"].get("confidence", 0.0)
        color = "green" if conf >= 0.7 else "yellow" if conf >= 0.4 else "red"
        st.markdown(
            f"<span style='background-color:{color};color:white;padding:4px;border-radius:3px'>"
            f"{decision['decision'].get('action')} ({conf:.2f})</span> - {decision['decision'].get('reason','')}",
            unsafe_allow_html=True,
        )
    else:
        st.write("No decision logged yet.")

    agent_info = get_last_agent_decision()
    show_agent_status(agent_info, auto_mode)

    portfolio_tabs = st.tabs(["Simulated Portfolio", "Real Holdings", "Last Trades", "Trade History Summary"])
    with portfolio_tabs[0]:
        if sim_data:
            show_portfolio_table(sim_data.get("positions", []), "Simulated Holdings")
            show_sim_summary(sim_data.get("summary", {}), sim_data.get("balance", 0.0))
        else:
            st.info("Simulation mode disabled or no data available.")
        st.caption(f"Last Updated: {last_updated} UTC")

    with portfolio_tabs[1]:
        tabs = ["Crypto", "Stocks"] + (["Forex"] if forex_enabled else [])
        real_tabs = st.tabs(tabs)
        with real_tabs[0]:
            show_portfolio_table(real_holdings.get("crypto", []), "Crypto Account Holdings")
        with real_tabs[1]:
            show_portfolio_table(real_holdings.get("stocks", []), "Stock Account Holdings")
        if forex_enabled and len(real_tabs) > 2:
            with real_tabs[2]:
                show_portfolio_table(real_holdings.get("forex", []), "Forex Account Holdings")
        st.caption(f"Last Updated: {last_updated} UTC")

    with portfolio_tabs[2]:
        last_trades = get_last_trade_per_market()
        markets = ["crypto", "stocks"] + (["forex"] if forex_enabled else [])
        for market_label in markets:
            trade = last_trades.get(market_label)
            st.subheader(market_label.capitalize())
            if trade:
                st.write(
                    f"{trade['timestamp']} {trade['symbol']} {trade['side']} "
                    f"{trade['quantity']} @ {trade['price']} confidence={trade.get('reason','')}"
                )
            else:
                st.write("No trades yet.")
        st.caption(f"Last Updated: {last_updated} UTC")

    with portfolio_tabs[3]:
        show_trade_history(trade_history)
        show_performance_view(metrics)
        st.caption(f"Last Updated: {last_updated} UTC")

    crypto_data = get_live_chart_data(crypto_symbol, timeframe=timeframe)
    stock_data = get_live_chart_data(stock_symbol, timeframe=timeframe)
    forex_data = get_live_chart_data(forex_symbol, timeframe=timeframe) if forex_enabled else []

    chart_tabs = ["Crypto Chart", "Stocks Chart"] + (["Forex Chart"] if forex_enabled else [])
    chart_tabs.extend(["Equity Curve", "AI Feed", "Heatmap", "Logs"])
    log_tabs = st.tabs(chart_tabs)
    idx = 0
    with log_tabs[idx]:
        show_crypto_view(crypto_data, timeframe=timeframe)
    idx += 1
    with log_tabs[idx]:
        show_stocks_view(stock_data, timeframe=timeframe)
    idx += 1
    if forex_enabled:
        with log_tabs[idx]:
            show_forex_view(forex_data, timeframe=timeframe)
        idx += 1
    with log_tabs[idx]:
        show_equity_curve(equity_curve_data)
    idx += 1
    with log_tabs[idx]:
        show_ai_thought_feed(ai_feed)
    idx += 1
    with log_tabs[idx]:
        show_conviction_heatmap(sentiment)
    idx += 1
    with log_tabs[idx]:
        show_log_view(logs)

    if sentiment:
        st.sidebar.markdown("### Sentiment Scores")
        st.sidebar.write("Sources: " + ", ".join(sorted(sentiment.keys())))
        st.sidebar.json(sentiment)


if __name__ == "__main__":
    main()
