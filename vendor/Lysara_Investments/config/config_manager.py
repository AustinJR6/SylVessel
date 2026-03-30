import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from utils.runtime_paths import runtime_path


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_alpaca_base_url(value: str | None) -> str:
    base = (value or "https://paper-api.alpaca.markets").strip().rstrip("/")
    if base.endswith("/v2"):
        base = base[:-3]
    return base


class ConfigManager:
    """Load runtime config from .env and local JSON settings files."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent

    def _settings_path(self, market: str) -> Path:
        clean_market = str(market or "").strip().lower()
        mapping = {
            "crypto": "settings_crypto.json",
            "stocks": "settings_stocks.json",
            "forex": "settings_forex.json",
        }
        filename = mapping.get(clean_market)
        if not filename:
            raise ValueError(f"Unsupported market: {market}")
        return self.base_dir / "config" / filename

    def load_market_settings(self, market: str) -> dict:
        return _load_json(self._settings_path(market))

    def save_market_settings(self, market: str, payload: dict) -> dict:
        path = self._settings_path(market)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = payload if isinstance(payload, dict) else {}
        path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2), encoding="utf-8")
        return normalized

    def load_config(self) -> dict:
        load_dotenv(self.base_dir / ".env")
        env = os.environ

        crypto_settings = _load_json(self.base_dir / "config" / "settings_crypto.json")
        stocks_settings = _load_json(self.base_dir / "config" / "settings_stocks.json")
        forex_settings = _load_json(self.base_dir / "config" / "settings_forex.json")

        robinhood_private_key = (
            env.get("ROBINHOOD_PRIVATE_KEY")
            or env.get("ROBINHOOD_SIGNING_KEY")
            or env.get("ROBINHOOD_PUBLIC_KEY")
            or ""
        )

        config = {
            "ENABLE_REDDIT_SENTIMENT": _as_bool(env.get("ENABLE_REDDIT_SENTIMENT", "false"), False),
            "simulation_mode": _as_bool(env.get("SIMULATION_MODE", "true"), True),
            "LIVE_TRADING_ENABLED": _as_bool(env.get("LIVE_TRADING_ENABLED", "true"), True),
            "ENABLE_CRYPTO_TRADING": _as_bool(env.get("ENABLE_CRYPTO_TRADING", "true"), True),
            "ENABLE_STOCK_TRADING": _as_bool(env.get("ENABLE_STOCK_TRADING", "true"), True),
            "FOREX_ENABLED": _as_bool(env.get("FOREX_ENABLED", "false"), False),
            "ENABLE_AI_STRATEGY": _as_bool(env.get("ENABLE_AI_STRATEGY", "true"), True),
            "ENABLE_AI_ASSET_DISCOVERY": _as_bool(env.get("ENABLE_AI_ASSET_DISCOVERY", "false"), False),
            "ENABLE_AI_TRADE_EXECUTION": _as_bool(env.get("ENABLE_AI_TRADE_EXECUTION", "false"), False),
            "SHOW_MANUAL_TRADING_UI": _as_bool(env.get("SHOW_MANUAL_TRADING_UI", "false"), False),
            "TRADE_SYMBOLS": [
                s.strip().upper()
                for s in (env.get("TRADE_SYMBOLS", "") or "").replace(";", ",").split(",")
                if s.strip()
            ] or crypto_settings.get("trade_symbols", ["BTC-USD", "ETH-USD"]),
            "starting_balance": float(env.get("STARTING_BALANCE", "1000")),
            "override_ttl_minutes": int(env.get("OVERRIDE_DEFAULT_TTL_MINUTES", "15")),
            "override_allowed_controls": [
                s.strip()
                for s in env.get(
                    "OVERRIDE_ALLOWED_CONTROLS",
                    "trade_cooldown,confidence_minimum,approval_requirement,event_risk_warning,sentiment_threshold",
                ).split(",")
                if s.strip()
            ],
            "ops_max_single_position_pct": float(
                env.get(
                    "OPS_MAX_SINGLE_POSITION_PCT",
                    str(float(crypto_settings.get("max_position_value_pct", 0.2)) * 100.0),
                )
            ),
            "ops_max_total_gross_exposure_pct": float(env.get("OPS_MAX_TOTAL_GROSS_EXPOSURE_PCT", "100")),
            "position_sizing_max_kelly_fraction": float(env.get("POSITION_SIZING_MAX_KELLY_FRACTION", "0.25")),
            "position_sizing_min_history_trades": int(env.get("POSITION_SIZING_MIN_HISTORY_TRADES", "8")),
            "event_risk_file": env.get("EVENT_RISK_FILE", str(runtime_path("event_risk_cache.json"))),
            "event_risk_lookahead_hours": int(env.get("EVENT_RISK_LOOKAHEAD_HOURS", "24")),
            "event_risk_warning_threshold": float(env.get("EVENT_RISK_WARNING_THRESHOLD", "0.45")),
            "event_risk_block_threshold": float(env.get("EVENT_RISK_BLOCK_THRESHOLD", "0.70")),
            "event_risk_reduction_threshold": float(env.get("EVENT_RISK_REDUCTION_THRESHOLD", "0.85")),
            "event_risk_reduction_factor": float(env.get("EVENT_RISK_REDUCTION_FACTOR", "0.50")),
            "sentiment_loop_interval_seconds": int(env.get("SENTIMENT_LOOP_INTERVAL_SECONDS", "300")),
            "event_risk_loop_interval_seconds": int(env.get("EVENT_RISK_LOOP_INTERVAL_SECONDS", "300")),
            "db_path": env.get("DB_PATH", str(runtime_path("trades.db"))),
            "sim_state_file": env.get("SIM_STATE_FILE", str(runtime_path("sim_state.json"))),
            "log_level": env.get("LOG_LEVEL", "INFO"),
            "log_file_path": env.get("LOG_FILE_PATH", str(runtime_path("logs", "trading_bot.log"))),
            "reddit_subreddits": [
                s.strip()
                for s in env.get("REDDIT_SUBREDDITS", "").split(",")
                if s.strip()
            ],
            "TECH_WEIGHT": float(env.get("TECH_WEIGHT", "0.5")),
            "SENT_WEIGHT": float(env.get("SENT_WEIGHT", "0.3")),
            "MARKET_WEIGHT": float(env.get("MARKET_WEIGHT", "0.2")),
            "REDDIT_SUBS": [
                s.strip()
                for s in env.get("REDDIT_SUBS", "").split(",")
                if s.strip()
            ],
            "NEWSAPI_KEY": env.get("NEWSAPI_KEY", ""),
            "api_keys": {
                "openai": env.get("OPENAI_API_KEY", ""),
                "newsapi": env.get("NEWSAPI_KEY", ""),
                "reddit_client_id": env.get("REDDIT_CLIENT_ID", ""),
                "reddit_secret": env.get("REDDIT_CLIENT_SECRET", "") or env.get("REDDIT_SECRET", ""),
                "reddit_user_agent": env.get("REDDIT_USER_AGENT", "LysaraSentimentBot/1.0"),
                "cryptopanic": env.get("CRYPTOPANIC_KEY", ""),
                "cryptopanic_base_url": env.get("CRYPTOPANIC_API_BASE_URL", "https://cryptopanic.com/api/developer/v2"),
                "x_api_key": env.get("X_API_KEY", ""),
                "x_api_key_secret": env.get("X_API_KEY_SECRET", ""),
                "x_bearer_token": env.get("X_BEARER_TOKEN", ""),
                "x_client_id": env.get("X_CLIENT_ID", ""),
                "x_client_secret": env.get("X_CLIENT_SECRET", ""),
                "whale_alert_api_key": env.get("WHALE_ALERT_API_KEY", ""),
                "finnhub_api_key": env.get("FINNHUB_API_KEY", ""),
                "tradingeconomics_api_key": env.get("TRADINGECONOMICS_API_KEY", ""),
                "tradingeconomics_api_secret": env.get("TRADINGECONOMICS_API_SECRET", ""),
                "coinmarketcal_api_key": env.get("COINMARKETCAL_API_KEY", ""),
                "slack_webhook": env.get("SLACK_WEBHOOK_URL", ""),
                "robinhood_api_key": env.get("ROBINHOOD_API_KEY", ""),
                "robinhood_private_key": robinhood_private_key,
                "robinhood_public_key": env.get("ROBINHOOD_PUBLIC_KEY", ""),
                "robinhood_base_url": env.get("ROBINHOOD_CRYPTO_API_BASE_URL", "https://trading.robinhood.com"),
                "coinbase": env.get("COINBASE_API_KEY", ""),
                "coinbase_secret": env.get("COINBASE_SECRET_KEY", ""),
                "alpaca": env.get("ALPACA_API_KEY", "") or env.get("ALPACA_KEY_ID", ""),
                "alpaca_secret": env.get("ALPACA_SECRET_KEY", ""),
                "alpaca_base_url": _normalize_alpaca_base_url(env.get("ALPACA_BASE_URL")),
                "oanda": env.get("OANDA_API_KEY", ""),
                "oanda_account_id": env.get("OANDA_ACCOUNT_ID", ""),
            },
            "crypto_settings": crypto_settings,
            "stocks_settings": stocks_settings,
            "forex_settings": forex_settings,
        }

        # Keep per-strategy configs aligned with global runtime mode.
        config["crypto_settings"]["simulation_mode"] = config["simulation_mode"]
        config["stocks_settings"]["simulation_mode"] = config["simulation_mode"]
        config["forex_settings"]["simulation_mode"] = config["simulation_mode"]
        if not config["ENABLE_REDDIT_SENTIMENT"]:
            config["reddit_subreddits"] = []
            config["REDDIT_SUBS"] = []

        return config
