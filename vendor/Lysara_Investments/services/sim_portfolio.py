import json
import logging
from pathlib import Path
from datetime import datetime

from data.price_cache import get_price
from utils.runtime_paths import env_or_runtime_path

class SimulatedPortfolio:
    """Simple portfolio tracker for simulation mode."""

    def __init__(
        self,
        starting_balance: float = 1000.0,
        state_file: str | None = None,
        trades_file: str | None = None,
    ):
        self.state_file = Path(state_file) if state_file else env_or_runtime_path("SIM_STATE_FILE", "sim_state.json")
        self.trades_file = Path(trades_file) if trades_file else env_or_runtime_path("SIM_TRADES_FILE", "simulated_trades.json")
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.open_positions: dict[str, float] = {}
        self.trade_history: list[dict] = []
        self._load_state()

    def _load_state(self):
        if self.state_file.is_file():
            try:
                data = json.loads(self.state_file.read_text())
                self.starting_balance = data.get("starting_balance", self.starting_balance)
                self.current_balance = data.get("current_balance", self.starting_balance)
                self.open_positions = data.get("open_positions", {})
            except Exception as e:
                logging.error(f"Failed to load simulation state: {e}")
        else:
            logging.info("No existing simulation state found; starting fresh.")

        if self.trades_file.is_file():
            try:
                self.trade_history = json.loads(self.trades_file.read_text())
            except Exception as e:
                logging.error(f"Failed to load trade history: {e}")
        else:
            self.trade_history = []

    def _save_state(self):
        data = {
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "open_positions": self.open_positions,
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data, indent=2))
            self.trades_file.write_text(json.dumps(self.trade_history, indent=2))
        except Exception as e:
            logging.error(f"Failed to save simulation state: {e}")

    def account_snapshot(self) -> dict:
        holdings_value = 0.0
        for asset, quantity in self.open_positions.items():
            cached = get_price(asset)
            price = float((cached or {}).get("price") or 0.0)
            if price > 0:
                holdings_value += float(quantity) * price
        return {
            "currency": "USD",
            "starting_balance": float(self.starting_balance),
            "balance": float(self.current_balance),
            "cash": float(self.current_balance),
            "buying_power": float(max(self.current_balance, 0.0)),
            "portfolio_value": float(self.current_balance + holdings_value),
        }

    def holdings_snapshot(self) -> dict[str, float]:
        return {
            str(asset): float(quantity)
            for asset, quantity in self.open_positions.items()
            if abs(float(quantity)) > 1e-8
        }

    def execute_trade(self, asset: str, action: str, size: float, price: float, confidence: float = 0.0):
        asset = str(asset or "").strip().upper()
        action = str(action or "").strip().lower()
        size = float(size or 0.0)
        price = float(price or 0.0)
        if not asset:
            return {"status": "rejected", "reason": "missing_asset"}
        if action not in {"buy", "sell"}:
            return {"status": "rejected", "reason": "unsupported_action"}
        if size <= 0 or price <= 0:
            return {"status": "rejected", "reason": "invalid_trade_size_or_price"}

        cost = size * price
        if action == "buy":
            if cost > self.current_balance:
                return {
                    "status": "rejected",
                    "reason": "insufficient_funds",
                    "required_cash": cost,
                    "available_cash": self.current_balance,
                }
            self.current_balance -= cost
            self.open_positions[asset] = self.open_positions.get(asset, 0.0) + size
        else:
            current_position = self.open_positions.get(asset, 0.0)
            if size > current_position:
                return {
                    "status": "rejected",
                    "reason": "insufficient_position",
                    "requested_size": size,
                    "available_size": current_position,
                }
            self.current_balance += cost
            self.open_positions[asset] = current_position - size
            if abs(self.open_positions[asset]) < 1e-8:
                self.open_positions.pop(asset, None)

        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "asset": asset,
            "action": action,
            "size": size,
            "price": price,
            "confidence": confidence,
        }
        self.trade_history.append(trade)
        self._save_state()
        logging.info(
            f"[SIM] Executed {action.upper()} {size} {asset} @ ${price} | New balance: ${self.current_balance:.2f}"
        )
        return {
            "status": "filled",
            "asset": asset,
            "action": action,
            "size": size,
            "price": price,
            "balance": self.current_balance,
        }

    def reset(self, starting_balance: float | None = None):
        """Clear simulation state and trade history."""
        if starting_balance is not None:
            self.starting_balance = float(starting_balance)
        self.current_balance = self.starting_balance
        self.open_positions = {}
        self.trade_history = []
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            if self.trades_file.exists():
                self.trades_file.unlink()
        except Exception as e:
            logging.error(f"Failed to remove simulation files: {e}")
        self._save_state()
        logging.info("Simulation state reset")
