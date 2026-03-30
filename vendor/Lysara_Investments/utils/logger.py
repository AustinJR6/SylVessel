# utils/logger.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(level: str = "INFO", log_file_path: str = "logs/trading_bot.log"):
    """
    Sets up global logging config. Outputs to both console and file.
    """
    Path("logs").mkdir(exist_ok=True)

    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    file_handler = RotatingFileHandler(
        log_file_path,
        mode="a",
        maxBytes=1024 * 1024,
        backupCount=5,
    )
    logging.basicConfig(
        level=level.upper(),
        format=log_format,
        handlers=[logging.StreamHandler(), file_handler],
    )

    logging.info("Logging initialized.")
