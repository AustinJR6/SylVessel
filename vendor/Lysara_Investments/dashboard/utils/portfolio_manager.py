import asyncio
import logging
from typing import List, Dict

from api.crypto_api import CryptoAPI
from services.alpaca_manager import AlpacaManager
from api.forex_api import ForexAPI
from services.sim_portfolio import SimulatedPortfolio


class PortfolioManager:
    """Helper to load live and simulated portfolio information."""

    def __init__(self, config: Dict):
        self.config = config or {}
        starting = self.config.get("starting_balance", 1000.0)
        state_file = self.config.get("sim_state_file", "data/sim_state.json")
        self.sim_portfolio = SimulatedPortfolio(
            starting_balance=starting, state_file=state_file
        )

    @staticmethod
    def _looks_placeholder(value: str | None) -> bool:
        v = (value or "").strip().lower()
        return not v or v.startswith("your_") or "placeholder" in v

    async def _fetch_live_holdings(self) -> List[Dict]:
        """Fetch holdings from available API integrations."""
        holdings: List[Dict] = []
        api_keys = self.config.get("api_keys", {})

        # Crypto holdings via Binance
        if not self._looks_placeholder(api_keys.get("binance")) and not self._looks_placeholder(
            api_keys.get("binance_secret")
        ):
            try:
                api = CryptoAPI(
                    api_key=api_keys.get("binance"),
                    secret_key=api_keys.get("binance_secret", ""),
                    simulation_mode=False,
                )
                data = await api.fetch_holdings()
                for asset, qty in data.items():
                    price = await api.fetch_market_price(f"{asset}-USD")
                    curr = float(price.get("price", 0))
                    holdings.append(
                        {
                            "asset": asset,
                            "quantity": qty,
                            "entry_price": None,
                            "current_price": curr,
                            "pnl": None,
                        }
                    )
                await api.close()
            except Exception as e:
                logging.error(f"Failed to fetch crypto holdings: {e}")

        # Stock holdings via Alpaca
        if not self._looks_placeholder(api_keys.get("alpaca")) and not self._looks_placeholder(
            api_keys.get("alpaca_secret")
        ):
            try:
                api = AlpacaManager(
                    api_key=api_keys.get("alpaca"),
                    api_secret=api_keys.get("alpaca_secret"),
                    base_url=api_keys.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
                    simulation_mode=False,
                )
                positions = await api.get_positions()
                for p in positions:
                    if isinstance(p, dict):
                        symbol = p.get("symbol", "")
                        qty = p.get("qty", 0)
                        entry = p.get("avg_entry_price", 0)
                        current = p.get("current_price", 0)
                        pnl = p.get("unrealized_pl", 0)
                    else:
                        symbol = getattr(p, "symbol", "")
                        qty = getattr(p, "qty", 0)
                        entry = getattr(p, "avg_entry_price", 0)
                        current = getattr(p, "current_price", 0)
                        pnl = getattr(p, "unrealized_pl", 0)
                    holdings.append(
                        {
                            "asset": symbol,
                            "quantity": float(qty or 0),
                            "entry_price": float(entry or 0),
                            "current_price": float(current or 0),
                            "pnl": float(pnl or 0),
                        }
                    )
            except Exception as e:
                logging.error(f"Failed to fetch stock holdings: {e}")

        # Placeholder for forex holdings
        if self.config.get("FOREX_ENABLED", False) and api_keys.get("oanda") and api_keys.get("oanda_account_id"):
            try:
                api = ForexAPI(
                    api_key=api_keys.get("oanda"),
                    account_id=api_keys.get("oanda_account_id"),
                    simulation_mode=False,
                )
                info = await api.get_account_info()
                balance = float(info.get("balance", 0))
                holdings.append(
                    {
                        "asset": "Forex Account",
                        "quantity": balance,
                        "entry_price": None,
                        "current_price": balance,
                        "pnl": None,
                    }
                )
                await api.close()
            except Exception as e:
                logging.error(f"Failed to fetch forex holdings: {e}")

        return holdings

    async def _fetch_crypto_holdings(self) -> List[Dict]:
        """Fetch crypto holdings via Binance."""
        holdings: List[Dict] = []
        api_keys = self.config.get("api_keys", {})
        if self._looks_placeholder(api_keys.get("binance")) or self._looks_placeholder(
            api_keys.get("binance_secret")
        ):
            return holdings
        try:
            api = CryptoAPI(
                api_key=api_keys.get("binance"),
                secret_key=api_keys.get("binance_secret", ""),
                simulation_mode=False,
            )
            data = await api.fetch_holdings()
            for asset, qty in data.items():
                price = await api.fetch_market_price(f"{asset}-USD")
                curr = float(price.get("price", 0))
                holdings.append(
                    {
                        "asset": asset,
                        "quantity": qty,
                        "entry_price": None,
                        "current_price": curr,
                        "pnl": None,
                    }
                )
            await api.close()
        except Exception as e:
            logging.error(f"Failed to fetch crypto holdings: {e}")
        return holdings

    async def _fetch_stock_holdings(self) -> List[Dict]:
        """Fetch stock holdings via Alpaca."""
        holdings: List[Dict] = []
        api_keys = self.config.get("api_keys", {})
        if self._looks_placeholder(api_keys.get("alpaca")) or self._looks_placeholder(
            api_keys.get("alpaca_secret")
        ):
            return holdings
        try:
            api = AlpacaManager(
                api_key=api_keys.get("alpaca"),
                api_secret=api_keys.get("alpaca_secret"),
                base_url=api_keys.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
                simulation_mode=False,
            )
            positions = await api.get_positions()
            for p in positions:
                if isinstance(p, dict):
                    symbol = p.get("symbol", "")
                    qty = p.get("qty", 0)
                    entry = p.get("avg_entry_price", 0)
                    current = p.get("current_price", 0)
                    pnl = p.get("unrealized_pl", 0)
                else:
                    symbol = getattr(p, "symbol", "")
                    qty = getattr(p, "qty", 0)
                    entry = getattr(p, "avg_entry_price", 0)
                    current = getattr(p, "current_price", 0)
                    pnl = getattr(p, "unrealized_pl", 0)
                holdings.append(
                    {
                        "asset": symbol,
                        "quantity": float(qty or 0),
                        "entry_price": float(entry or 0),
                        "current_price": float(current or 0),
                        "pnl": float(pnl or 0),
                    }
                )
        except Exception as e:
            logging.error(f"Failed to fetch stock holdings: {e}")
        return holdings

    async def _fetch_forex_holdings(self) -> List[Dict]:
        """Fetch forex account balance via OANDA."""
        holdings: List[Dict] = []
        api_keys = self.config.get("api_keys", {})
        if not (api_keys.get("oanda") and api_keys.get("oanda_account_id")):
            return holdings
        try:
            api = ForexAPI(
                api_key=api_keys.get("oanda"),
                account_id=api_keys.get("oanda_account_id"),
                simulation_mode=False,
            )
            info = await api.get_account_info()
            balance = float(info.get("balance", 0))
            holdings.append(
                {
                    "asset": "Forex Account",
                    "quantity": balance,
                    "entry_price": None,
                    "current_price": balance,
                    "pnl": None,
                }
            )
            await api.close()
        except Exception as e:
            logging.error(f"Failed to fetch forex holdings: {e}")
        return holdings

    async def _fetch_all_holdings(self) -> Dict[str, List[Dict]]:
        """Fetch holdings for all asset classes separately."""
        crypto = await self._fetch_crypto_holdings()
        stocks = await self._fetch_stock_holdings()
        forex = await self._fetch_forex_holdings() if self.config.get("FOREX_ENABLED", False) else []
        return {"crypto": crypto, "stocks": stocks, "forex": forex}

    def get_live_holdings(self) -> List[Dict]:
        try:
            return asyncio.run(self._fetch_live_holdings())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._fetch_live_holdings())
            loop.close()
            return result

    def get_account_holdings(self) -> Dict[str, List[Dict]]:
        """Return real holdings for crypto, stocks and forex separately."""
        try:
            return asyncio.run(self._fetch_all_holdings())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._fetch_all_holdings())
            loop.close()
            return result

    def get_simulated_portfolio(self) -> Dict:
        """Load simulated portfolio state from file."""
        self.sim_portfolio._load_state()
        positions = []
        for asset, qty in self.sim_portfolio.open_positions.items():
            positions.append(
                {
                    "asset": asset,
                    "quantity": qty,
                    "entry_price": None,
                    "current_price": 0.0,
                    "pnl": None,
                }
            )

        trades = self.sim_portfolio.trade_history
        closed = [t for t in trades if t.get("pnl") is not None]
        wins = [t for t in closed if t.get("pnl", 0) > 0]
        win_rate = round(len(wins) / len(closed) * 100, 2) if closed else 0.0
        avg_return = (
            round(sum(t.get("pnl", 0) for t in closed) / len(closed), 4)
            if closed
            else 0.0
        )

        summary = {
            "win_rate": win_rate,
            "avg_return": avg_return,
            "trade_count": len(trades),
        }

        return {
            "balance": self.sim_portfolio.current_balance,
            "positions": positions,
            "trades": trades,
            "summary": summary,
        }
