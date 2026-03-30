# utils/logger.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .runtime_paths import env_or_runtime_path

def setup_logging(level: str = "INFO", log_file_path: str | None = None):
    """
    Sets up global logging config. Outputs to both console and file.
    """
    resolved_log_path = Path(log_file_path) if log_file_path else env_or_runtime_path("LOG_FILE_PATH", "logs", "trading_bot.log")
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    file_handler = RotatingFileHandler(
        resolved_log_path,
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
