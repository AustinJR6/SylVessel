from __future__ import annotations

import logging
import sys

from core.config_loader import config


def configure_logging() -> None:
    level = getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
