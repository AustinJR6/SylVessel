# services/bot_launcher.py

import asyncio
import logging
import os


async def _run_strategy_rebalance_loop(meta_router, strategy_instances, interval_seconds=3600):
    """Hourly: update MetaStrategyRouter with performance data, adjust position size caps."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            for strat in strategy_instances:
                name = strat.__class__.__name__
                sharpe = getattr(strat, "sharpe_ratio", 0.5)
                win_rate = getattr(strat, "win_rate", 0.5)
                total_return = getattr(strat, "total_return", 0.0)
                if hasattr(meta_router, "update_performance"):
                    meta_router.update_performance(name, sharpe, win_rate, total_return)
            logging.info("[MetaRouter] Rebalance tick complete")
        except Exception as e:
            logging.error(f"[MetaRouter] Rebalance error: {e}")

from api.crypto_api import CryptoAPI
from api.forex_api import ForexAPI
from risk.risk_manager import RiskManager
from strategies.crypto.momentum import MomentumStrategy
from data.market_data_crypto import start_crypto_market_feed
from data.market_data_coingecko import start_coingecko_polling
from data.market_data_alpaca import start_stock_ws_feed
from db.db_manager import DatabaseManager
from services.background_tasks import BackgroundTasks
from services.sim_portfolio import SimulatedPortfolio
from services.heartbeat import heartbeat
from services.daemon_state import get_state


def _is_crypto_symbol(symbol: str) -> bool:
    token = str(symbol or "").strip().upper()
    return bool(token) and "-" in token


def _filter_crypto_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    filtered: list[str] = []
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not _is_crypto_symbol(symbol) or symbol in seen:
            continue
        seen.add(symbol)
        filtered.append(symbol)
    return filtered


class BotLauncher:
    def __init__(self, config: dict):
        self.config = config
        env_syms = os.getenv("TRADE_SYMBOLS", "")
        if env_syms:
            syms = [s.strip().upper() for s in env_syms.split(";") if s.strip()] if ";" in env_syms else [s.strip().upper() for s in env_syms.split(",") if s.strip()]
            crypto = [s for s in syms if "-" in s]
            stocks = [s for s in syms if "-" not in s]
            if crypto:
                self.config["TRADE_SYMBOLS"] = crypto
            if stocks:
                self.config.setdefault("stocks_settings", {}).setdefault(
                    "trade_symbols", stocks
                )
        self.db = DatabaseManager(config.get("db_path", "trades.db"))
        state = get_state()
        state.register_db_manager(self.db)
        state.set_runtime_config(config)
        self.bg_tasks = BackgroundTasks(self.config)
        self.sim_portfolio = None
        if self.config.get("simulation_mode", True):
            starting = self.config.get("starting_balance", 1000.0)
            state_file = self.config.get(
                "sim_state_file", "data/sim_state.json"
            )
            self.sim_portfolio = SimulatedPortfolio(
                starting_balance=starting, state_file=state_file
            )
            state.register_sim_portfolio(self.sim_portfolio)

    def start_all_bots(self):
        asyncio.create_task(self.bg_tasks.run_sentiment_loop())
        asyncio.create_task(self.bg_tasks.run_event_risk_loop())
        asyncio.create_task(heartbeat())

        if self.config.get("ENABLE_CRYPTO_TRADING", True):
            asyncio.create_task(self.start_crypto_bots())

        if self.config.get("ENABLE_STOCK_TRADING", False):
            asyncio.create_task(self.start_stock_bots())

        if self.config.get("FOREX_ENABLED", False):
            api_keys = self.config.get("api_keys", {})
            if api_keys.get("oanda") and api_keys.get("oanda_account_id"):
                asyncio.create_task(self.start_forex_bots())
            else:
                logging.warning("FOREX_ENABLED but OANDA credentials missing. Forex bots disabled.")

    async def start_crypto_bots(self):
        """Launch one or more crypto strategy instances."""
        logging.info(" Starting crypto bots...")

        api_keys = self.config["api_keys"]
        settings = self.config.get("crypto_settings", {})
        base_symbols_env = self.config.get("TRADE_SYMBOLS")
        base_symbols = (
            base_symbols_env
            if base_symbols_env
            else settings.get("trade_symbols", ["BTC-USD", "ETH-USD"])
        )

        extra_symbols: list[str] = []
        if self.config.get("ENABLE_AI_ASSET_DISCOVERY", False):
            from services.ai_strategist import ai_discover_assets

            try:
                extra_symbols = await ai_discover_assets(base_symbols)
            except Exception as e:
                logging.error(f"AI asset discovery failed: {e}")

        extra_symbols = _filter_crypto_symbols(extra_symbols)
        all_symbols = _filter_crypto_symbols(base_symbols + extra_symbols)

        crypto_api = CryptoAPI(
            api_key=api_keys.get("robinhood_api_key", ""),
            secret_key=api_keys.get("robinhood_private_key", "") or api_keys.get("robinhood_public_key", ""),
            simulation_mode=self.config.get("simulation_mode", True),
            portfolio=self.sim_portfolio,
            config=self.config,
        )

        account_info = await crypto_api.fetch_account_info()
        if account_info.get("error"):
            logging.error("Robinhood crypto API unavailable: %s", account_info.get("message"))
            get_state().set_broker_health("crypto", False, "crypto_api_unavailable")
            await crypto_api.close()
            return
        get_state().register_api(crypto_api, "crypto")
        get_state().set_broker_health("crypto", True, "crypto_api_ready")

        asyncio.create_task(start_crypto_market_feed(all_symbols, crypto_api))
        asyncio.create_task(start_coingecko_polling(all_symbols))

        # A4 — Price trigger service background task
        from services.price_trigger_service import init_trigger_service
        trigger_svc = init_trigger_service(
            crypto_api=crypto_api,
            sim_portfolio=self.sim_portfolio,
            simulation_mode=self.config.get("simulation_mode", True),
        )
        asyncio.create_task(trigger_svc.run_trigger_loop())
        logging.info("[BotLauncher] Price trigger loop started")

        # A5 — Daily feed background task
        from data.market_data_daily import run_daily_feed
        all_crypto_symbols = list(all_symbols)
        asyncio.create_task(run_daily_feed(all_crypto_symbols, interval_seconds=3600))
        logging.info(f"[BotLauncher] Daily feed started for {len(all_crypto_symbols)} symbols")

        # A3 — Position registry
        from services.position_registry import get_registry
        registry = get_registry()

        # A2 — Symbol strategy map from config
        symbol_strategy_map = self.config.get("symbol_strategy_map", {})

        all_strategy_instances = []
        strategy_cfgs = settings.get("strategies") or [settings]
        for cfg in strategy_cfgs:
            sym_list = cfg.get("trade_symbols", base_symbols)
            cfg_full = {
                **settings,
                **cfg,
                "simulation_mode": self.config.get("simulation_mode", True),
                "LIVE_TRADING_ENABLED": self.config.get("LIVE_TRADING_ENABLED", True),
                "ENABLE_AI_TRADE_EXECUTION": self.config.get("ENABLE_AI_TRADE_EXECUTION", False),
            }

            # A1 — Extended strategy dispatch
            strat_type = cfg.get("type", "momentum").lower()
            if strat_type == "mean_reversion":
                from strategies.crypto.mean_reversion import MeanReversionStrategy as StratCls
            elif strat_type == "micro_scalping":
                from strategies.crypto.micro_scalping import MicroScalpingStrategy as StratCls
            elif strat_type == "pairs_trading":
                from strategies.crypto.pairs_trading import PairsTradingStrategy as StratCls
            elif strat_type == "ai_momentum_fusion":
                from strategies.ai_momentum_fusion import AIMomentumFusion as StratCls
            elif strat_type == "crypto_scalper":
                from strategies.crypto_scalper import CryptoScalper as StratCls
            elif strat_type in ("swing", "swingtradingstrategy"):
                from strategies.crypto.swing_trading import SwingTradingStrategy as StratCls
            else:
                StratCls = MomentumStrategy

            risk = RiskManager(crypto_api, cfg_full)
            await risk.update_equity()
            get_state().register_risk_manager(risk, "crypto")

            # A3 — Try to pass position_registry; fall back if constructor doesn't accept it
            try:
                strategy = StratCls(
                    api=crypto_api,
                    risk=risk,
                    config={**cfg_full, "market": "crypto"},
                    db=self.db,
                    symbol_list=sym_list,
                    sentiment_source=self.bg_tasks,
                    ai_symbols=extra_symbols,
                    position_registry=registry,
                )
            except TypeError:
                strategy = StratCls(
                    api=crypto_api,
                    risk=risk,
                    config={**cfg_full, "market": "crypto"},
                    db=self.db,
                    symbol_list=sym_list,
                    sentiment_source=self.bg_tasks,
                    ai_symbols=extra_symbols,
                )

            if not getattr(strategy, "position_registry", None):
                strategy.position_registry = registry
            if not getattr(strategy, "symbol", None) and sym_list:
                strategy.symbol = sym_list[0]
            get_state().register_strategy("crypto", strategy.__class__.__name__, sym_list, cfg_full, instance=strategy)
            all_strategy_instances.append(strategy)

        # A2 — Apply symbol_strategy_map: only start strategies assigned to their symbol
        for strategy_instance in all_strategy_instances:
            if symbol_strategy_map:
                sym = getattr(strategy_instance, "symbol", None)
                assigned = symbol_strategy_map.get(sym)
                if assigned and strategy_instance.__class__.__name__.lower() != assigned.lower():
                    logging.info(
                        f"[BotLauncher] Skipping {strategy_instance.__class__.__name__} for {sym} — assigned to {assigned}"
                    )
                    continue
            asyncio.create_task(strategy_instance.run())

        # A6 — MetaStrategyRouter rebalance loop
        from services.meta_strategy_router import MetaStrategyRouter
        meta_router = MetaStrategyRouter()
        asyncio.create_task(_run_strategy_rebalance_loop(meta_router, all_strategy_instances))

    async def start_stock_bots(self):
        logging.info("Starting stock bots...")

        api_keys = self.config.get("api_keys", {})
        settings = self.config.get("stocks_settings", {})
        base_symbols = settings.get("trade_symbols", ["AAPL", "TSLA"])

        alpaca_key = api_keys.get("alpaca")
        alpaca_secret = api_keys.get("alpaca_secret")
        base_url = api_keys.get(
            "alpaca_base_url",
            "https://paper-api.alpaca.markets",
        )
        if not alpaca_key or not alpaca_secret:
            logging.error(
                "Alpaca API credentials missing. Stock bots disabled."
            )
            return

        from services.alpaca_manager import AlpacaManager

        stock_api = AlpacaManager(
            api_key=alpaca_key,
            api_secret=alpaca_secret,
            base_url=base_url,
            simulation_mode=self.config.get("simulation_mode", True),
            portfolio=self.sim_portfolio,
            config=self.config,
        )

        await stock_api.get_account()
        get_state().register_api(stock_api, "stocks")
        get_state().set_broker_health("stocks", True, "stock_api_ready")

        risk = RiskManager(stock_api, settings)
        await risk.update_equity()

        from data.market_data_stocks import start_stock_polling_loop

        asyncio.create_task(start_stock_polling_loop(base_symbols, stock_api))
        asyncio.create_task(
            start_stock_ws_feed(
                base_symbols,
                alpaca_key,
                alpaca_secret,
                base_url,
            )
        )

        strategy_cfgs = settings.get("strategies") or [settings]
        for cfg in strategy_cfgs:
            sym_list = cfg.get("trade_symbols", base_symbols)
            cfg_full = {
                **settings,
                **cfg,
                "simulation_mode": self.config.get("simulation_mode", True),
                "LIVE_TRADING_ENABLED": self.config.get("LIVE_TRADING_ENABLED", True),
                "ENABLE_AI_TRADE_EXECUTION": self.config.get("ENABLE_AI_TRADE_EXECUTION", False),
            }
            from strategies.stocks.stock_momentum import StockMomentumStrategy as StratCls
            risk = RiskManager(stock_api, cfg_full)
            await risk.update_equity()
            get_state().register_risk_manager(risk, "stocks")
            strategy = StratCls(
                api=stock_api,
                risk=risk,
                config={**cfg_full, "market": "stocks"},
                db=self.db,
                symbol_list=sym_list,
            )
            if not getattr(strategy, "symbol", None) and sym_list:
                strategy.symbol = sym_list[0]
            get_state().register_strategy("stocks", strategy.__class__.__name__, sym_list, cfg_full, instance=strategy)
            asyncio.create_task(strategy.run())

    async def start_forex_bots(self):
        logging.info("Starting forex bots...")

        api_keys = self.config.get("api_keys", {})
        settings = self.config.get("forex_settings", {})
        base_instruments = settings.get("trade_symbols", ["EUR_USD", "GBP_USD"])

        api_key = api_keys.get("oanda")
        account_id = api_keys.get("oanda_account_id")
        if not api_key or not account_id:
            logging.error(
                "OANDA API credentials missing or invalid. Bots disabled."
            )
            return

        forex_api = ForexAPI(
            api_key=api_key,
            account_id=account_id,
            simulation_mode=self.config.get("simulation_mode", True),
            portfolio=self.sim_portfolio,
        )

        await forex_api.get_account_info()
        get_state().register_api(forex_api, "forex")
        get_state().set_broker_health("forex", True, "forex_api_ready")

        risk = RiskManager(forex_api, settings)
        await risk.update_equity()

        from data.market_data_forex import start_forex_polling_loop

        asyncio.create_task(
            start_forex_polling_loop(
                base_instruments,
                api_key,
                account_id,
            )
        )

        strategy_cfgs = settings.get("strategies") or [settings]
        for cfg in strategy_cfgs:
            inst_list = cfg.get("trade_symbols", base_instruments)
            cfg_full = {
                **settings,
                **cfg,
                "simulation_mode": self.config.get("simulation_mode", True),
                "LIVE_TRADING_ENABLED": self.config.get("LIVE_TRADING_ENABLED", True),
                "ENABLE_AI_TRADE_EXECUTION": self.config.get("ENABLE_AI_TRADE_EXECUTION", False),
            }
            from strategies.forex.rsi_trend import ForexRSITrendStrategy as StratCls
            risk = RiskManager(forex_api, cfg_full)
            await risk.update_equity()
            get_state().register_risk_manager(risk, "forex")
            strategy = StratCls(
                api=forex_api,
                risk=risk,
                config={**cfg_full, "market": "forex"},
                db=self.db,
                symbol_list=inst_list,
            )
            if not getattr(strategy, "symbol", None) and inst_list:
                strategy.symbol = inst_list[0]
            get_state().register_strategy("forex", strategy.__class__.__name__, inst_list, cfg_full, instance=strategy)
            asyncio.create_task(strategy.run())
