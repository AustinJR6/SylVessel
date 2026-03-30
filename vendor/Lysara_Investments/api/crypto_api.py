"""CryptoAPI now powered by :class:`RobinhoodCryptoClient`."""

from api.robinhood_crypto_client import RobinhoodCryptoClient


class CryptoAPI(RobinhoodCryptoClient):
    """Historical alias for the active Robinhood crypto adapter."""

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        *,
        simulation_mode: bool = True,
        portfolio=None,
        config: dict | None = None,
        trade_cooldown: int = 30,
    ) -> None:
        config = config or {}
        api_cfg = config.get("api_keys", {})
        robinhood_api_key = api_key or api_cfg.get("robinhood_api_key", "")
        signing_key = (
            secret_key
            or api_cfg.get("robinhood_private_key", "")
            or api_cfg.get("robinhood_public_key", "")
        )
        base_url = api_cfg.get("robinhood_base_url", "https://trading.robinhood.com")

        super().__init__(
            api_key=robinhood_api_key,
            private_key=signing_key,
            base_url=base_url,
            simulation_mode=simulation_mode,
            portfolio=portfolio,
            config=config,
            trade_cooldown=trade_cooldown,
        )

    async def fetch_holdings(self) -> dict:
        """Alias for ``get_holdings`` for backward compatibility."""
        return await self.get_holdings()
