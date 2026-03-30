# risk/risk_manager.py

import logging
import asyncio
from utils.notifications import send_slack_message

class RiskManager:
    def __init__(self, api_client, config: dict):
        self.api = api_client
        self.config = config
        self.max_drawdown = config.get("max_drawdown", 0.2)
        self.max_daily_loss = config.get("max_daily_loss", -200)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.max_consec_losses = config.get("max_consec_losses", 5)
        self.drawdown_triggered = False
        self.daily_loss = 0.0
        self.consec_losses = 0
        self.last_equity = None
        self.start_equity = None
        self.webhook = config.get("api_keys", {}).get("slack_webhook")

    async def _alert(self, message: str):
        if self.webhook:
            await send_slack_message(self.webhook, message)

    async def update_equity(self):
        info = await self.api.fetch_account_info()
        if isinstance(info, dict) and "balance" in info:
            self.last_equity = float(info["balance"])
        elif "portfolio_value" in info:
            self.last_equity = float(info["portfolio_value"])
        else:
            logging.warning("RiskManager: Could not retrieve equity from API.")
        return self.last_equity

    def get_position_size(self, price: float) -> float:
        if not self.last_equity or self.risk_per_trade <= 0:
            return 0
        dollar_risk = self.last_equity * self.risk_per_trade
        return round(dollar_risk / price, 6)

    def record_loss(self, amount: float):
        self.daily_loss += amount
        self.consec_losses += 1
        if self.daily_loss <= self.max_daily_loss or self.consec_losses >= self.max_consec_losses:
            self.drawdown_triggered = True
            logging.warning("Drawdown or loss limit reached. Trading disabled.")
            asyncio.create_task(self._alert("🚨 Max drawdown or consecutive losses hit"))

    async def check_daily_loss(self) -> bool:
        prev = self.last_equity
        equity = await self.update_equity()
        if equity is None:
            return True
        if self.start_equity is None:
            self.start_equity = equity
        if prev is not None and equity < prev:
            self.daily_loss += equity - prev
        if (equity - self.start_equity) <= self.max_daily_loss:
            self.drawdown_triggered = True
            logging.warning("Daily loss limit exceeded. Trading halted for the day.")
            await self._alert("🚨 Daily loss limit hit")
        return not self.drawdown_triggered

    def reset_daily_risk(self):
        self.daily_loss = 0.0
        self.consec_losses = 0
        self.drawdown_triggered = False

    def reset_streak(self):
        self.consec_losses = 0

    def sentiment_lockout(self, score_delta: float, threshold: float = -0.5) -> bool:
        if score_delta <= threshold:
            logging.warning("Sentiment crash detected. Locking trading.")
            self.drawdown_triggered = True
            return True
        return False

    def volatility_lockout(self, move: float, threshold: float = 0.15) -> bool:
        if abs(move) >= threshold:
            logging.warning("Volatility lockout triggered.")
            self.drawdown_triggered = True
            return True
        return False
