# strategies/base_strategy.py

import abc
from typing import Iterable

from services.daemon_state import get_state

class BaseStrategy(abc.ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, api, risk, config, db, symbol_list):
        self.api = api
        self.risk = risk
        self.config = config
        self.db = db
        self.symbols = list(symbol_list)
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.market = str(config.get("market", "") or self.__class__.__module__.split(".")[1]).lower()
        self.state = get_state()
        self.position_registry = None

    def trading_enabled(self, symbol: str | None = None) -> bool:
        if self.state.is_paused():
            return False
        if not self.state.is_strategy_enabled(self.market, self.__class__.__name__):
            return False
        if symbol and not self.state.is_symbol_enabled(self.market, symbol):
            return False
        return True

    def set_symbols(self, symbols: Iterable[str]) -> None:
        normalized = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
        self.symbols = normalized
        next_history = {}
        for symbol in normalized:
            next_history[symbol] = list(self.price_history.get(symbol) or [])
        self.price_history = next_history

    def record_decision(
        self,
        symbol: str | None,
        action: str,
        status: str,
        reasons: list[str] | None = None,
        *,
        confidence: float | None = None,
        market: str | None = None,
    ) -> None:
        self.state.record_strategy_decision(
            market=market or self.market,
            strategy_name=self.__class__.__name__,
            symbol=symbol,
            action=action,
            status=status,
            reasons=[str(reason) for reason in (reasons or []) if str(reason).strip()],
            confidence=confidence,
        )

    def registry_allows(self, symbol: str, side: str) -> tuple[bool, list[str]]:
        registry = self.position_registry
        if registry is None:
            return True, []
        return registry.can_open(symbol, side, self.__class__.__name__)

    @abc.abstractmethod
    async def run(self):
        """
        Run the main strategy loop.
        """
        pass

    @abc.abstractmethod
    async def enter_trade(self, symbol: str, price: float, side: str):
        """
        Submit a new trade with direction and log it.
        """
        pass
