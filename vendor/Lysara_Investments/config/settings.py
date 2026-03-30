from pydantic import BaseModel, SecretStr, ValidationError
from typing import List, Optional
import json
import os
from dotenv import dotenv_values


class ApiKeys(BaseModel):
    openai: SecretStr
    newsapi: SecretStr
    reddit_client_id: SecretStr
    reddit_secret: SecretStr
    binance_read_key: SecretStr
    binance_read_secret: SecretStr
    binance_trade_key: Optional[SecretStr] = None
    binance_trade_secret: Optional[SecretStr] = None
    alpaca_key: SecretStr
    alpaca_secret: SecretStr
    oanda_key: Optional[SecretStr] = None
    slack_webhook: Optional[SecretStr] = None


class Flags(BaseModel):
    live_trading: bool = False
    markets_crypto: bool = True
    markets_stocks: bool = True
    markets_forex: bool = False
    use_ai: bool = True
    use_scanner: bool = True


class StrategyToggles(BaseModel):
    cooldown_seconds: int = 60
    ai_max_cpm: int = 10  # calls per minute per symbol


class Settings(BaseModel):
    api: ApiKeys
    flags: Flags = Flags()
    symbols_crypto: List[str] = []
    symbols_stocks: List[str] = []
    symbols_forex: List[str] = []
    strategies: StrategyToggles = StrategyToggles()


def load_settings(path_env: str = ".env", path_json: str = "config/runtime.json") -> Settings:
    env = dotenv_values(path_env)
    json_cfg = {}
    if os.path.exists(path_json):
        with open(path_json, "r") as f:
            json_cfg = json.load(f)

    def env_bool(name: str, default: bool):
        if name in env:
            return env.get(name) == "1"
        return json_cfg.get("flags", {}).get(name.lower(), default)

    merged = {
        "api": {
            "openai": env.get("OPENAI_API_KEY") or json_cfg.get("api", {}).get("openai"),
            "newsapi": env.get("NEWSAPI_KEY") or json_cfg.get("api", {}).get("newsapi"),
            "reddit_client_id": env.get("REDDIT_CLIENT_ID") or json_cfg.get("api", {}).get("reddit_client_id"),
            "reddit_secret": env.get("REDDIT_SECRET") or json_cfg.get("api", {}).get("reddit_secret"),
            "binance_read_key": env.get("BINANCE_API_KEY_READ") or json_cfg.get("api", {}).get("binance_read_key"),
            "binance_read_secret": env.get("BINANCE_API_SECRET_READ") or json_cfg.get("api", {}).get("binance_read_secret"),
            "binance_trade_key": env.get("BINANCE_API_KEY_TRADE") or json_cfg.get("api", {}).get("binance_trade_key"),
            "binance_trade_secret": env.get("BINANCE_API_SECRET_TRADE") or json_cfg.get("api", {}).get("binance_trade_secret"),
            "alpaca_key": env.get("ALPACA_KEY_ID") or json_cfg.get("api", {}).get("alpaca_key"),
            "alpaca_secret": env.get("ALPACA_SECRET_KEY") or json_cfg.get("api", {}).get("alpaca_secret"),
            "oanda_key": env.get("OANDA_API_KEY") or json_cfg.get("api", {}).get("oanda_key"),
            "slack_webhook": env.get("SLACK_WEBHOOK_URL") or json_cfg.get("api", {}).get("slack_webhook"),
        },
        "flags": {
            "live_trading": env_bool("LIVE_TRADING", False),
            "markets_crypto": env_bool("MARKETS_CRYPTO", True),
            "markets_stocks": env_bool("MARKETS_STOCKS", True),
            "markets_forex": env_bool("MARKETS_FOREX", False),
            "use_ai": env_bool("USE_AI", True),
            "use_scanner": env_bool("USE_SCANNER", True),
        },
        "symbols_crypto": json_cfg.get("symbols_crypto", []),
        "symbols_stocks": json_cfg.get("symbols_stocks", []),
        "symbols_forex": json_cfg.get("symbols_forex", []),
        "strategies": json_cfg.get("strategies", {}),
    }
    try:
        return Settings(**merged)
    except ValidationError as e:
        raise SystemExit(f"Invalid config: {e}")
