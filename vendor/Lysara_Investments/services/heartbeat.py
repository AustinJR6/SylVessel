import asyncio
import logging

async def heartbeat(interval: int = 60):
    """Periodic health check log."""
    while True:
        logging.debug("Heartbeat alive")
        await asyncio.sleep(interval)
