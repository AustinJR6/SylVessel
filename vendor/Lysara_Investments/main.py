# main.py

import argparse
import os
import sys
import asyncio
if sys.platform.startswith("win"):
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

import logging
from config.config_manager import ConfigManager
from utils.logger import setup_logging
from utils.guardrails import confirm_live_mode
from services.bot_launcher import BotLauncher
from services.daemon_state import get_state
from services.runtime_store import get_runtime_store

def _parse_args():
    parser = argparse.ArgumentParser(description="Run Lysara trading services")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--simulate", action="store_true", help="Force simulation mode")
    mode_group.add_argument("--live", action="store_true", help="Force live mode")
    parser.add_argument("--no-control-api", action="store_true", help="Disable the Tier 2 control API")
    return parser.parse_args()


async def _start_control_api(port: int):
    """Run the FastAPI control server as an asyncio task."""
    try:
        import uvicorn
        from services.control_api import app as control_app
        host = os.getenv("CONTROL_API_HOST", "127.0.0.1").strip() or "127.0.0.1"

        cfg = uvicorn.Config(
            control_app,
            host=host,
            port=port,
            log_level="warning",
            loop="none",       # reuse the existing event loop
            access_log=False,
        )
        server = uvicorn.Server(cfg)
        logging.info(f"Lysara Control API listening on {host}:{port}")
        await server.serve()
    except Exception as e:
        logging.error(f"Control API failed to start: {e}")


async def main():
    args = _parse_args()
    if args.simulate:
        os.environ["SIMULATION_MODE"] = "true"
    elif args.live:
        os.environ["SIMULATION_MODE"] = "false"

    config = ConfigManager().load_config()
    confirm_live_mode(config.get("simulation_mode", True))

    setup_logging(
        level=config.get("log_level", "INFO"),
        log_file_path=config.get("log_file_path", "logs/trading_bot.log")
    )

    logging.info("Lysara Investments booting up...")
    logging.info(
        f"Simulation mode: {config.get('simulation_mode', True)} | Risk per trade: {config.get('crypto_settings', {}).get('risk_per_trade')}"
    )

    # Populate daemon state so the control API has context
    state = get_state()
    state.simulation_mode = config.get("simulation_mode", True)
    state.autonomous_mode = bool(config.get("ENABLE_AI_TRADE_EXECUTION", False))
    state.set_runtime_config(config)
    runtime_overrides = get_runtime_store().get_runtime()
    state.apply_runtime_overrides(runtime_overrides)

    launcher = BotLauncher(config)
    launcher.start_all_bots()

    # Start Tier 2 Control API unless explicitly disabled
    if not args.no_control_api:
        port = int(os.getenv("CONTROL_API_PORT", "18791"))
        asyncio.create_task(_start_control_api(port))

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Shutdown requested. Exiting gracefully.")

