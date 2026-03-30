import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from onchain_agent.initialize_agent import initialize_agent


CHAIN_ID_TO_NETWORK_ID = {
    84532: "base-sepolia",
    8453: "base-mainnet",
    1: "ethereum-mainnet",
    11155111: "ethereum-sepolia",
}

def launch_agent(test: bool = False):
    """Load configuration and start the onchain agent."""

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    load_dotenv(root / "onchain-agent" / ".env.local")

    cdp_api_key_id = os.getenv("CDP_API_KEY_ID") or os.getenv("CDP_API_KEY")
    cdp_api_key_secret = (
        os.getenv("CDP_API_KEY_SECRET")
        or os.getenv("CDP_API_KEY_PRIVATE_KEY")
        or os.getenv("CDP_PROJECT_ID")
    )
    cdp_wallet_secret = os.getenv("CDP_WALLET_SECRET")
    chain_id = os.getenv("CHAIN_ID")
    network_id = os.getenv("NETWORK_ID")
    rpc_url = os.getenv("RPC_URL")

    missing = [
        name
        for name, val in [
            ("CDP_API_KEY_ID (or CDP_API_KEY)", cdp_api_key_id),
            ("CDP_API_KEY_SECRET (or CDP_API_KEY_PRIVATE_KEY)", cdp_api_key_secret),
            ("CDP_WALLET_SECRET", cdp_wallet_secret),
        ]
        if not val
    ]
    if missing:
        logging.error(f"Missing required env vars: {', '.join(missing)}")
        return

    chain_id_int = None
    try:
        if chain_id:
            chain_id_int = int(chain_id)
    except ValueError:
        logging.error("CHAIN_ID must be an integer when provided")
        return

    resolved_network_id = network_id or (
        CHAIN_ID_TO_NETWORK_ID.get(chain_id_int) if chain_id_int is not None else None
    )

    try:
        initialize_agent(
            cdp_api_key_id=cdp_api_key_id,
            cdp_api_key_secret=cdp_api_key_secret,
            cdp_wallet_secret=cdp_wallet_secret,
            chain_id=chain_id_int,
            rpc_url=rpc_url,
            network_id=resolved_network_id,
            test=test,
        )
    except Exception as exc:
        logging.error(f"Agent initialization failed: {exc}")
        return
