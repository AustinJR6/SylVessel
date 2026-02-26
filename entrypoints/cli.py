from __future__ import annotations

import logging

from core.brain import Brain
from core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    brain = Brain.create_default(mode="claude")

    print("SylVessel CLI")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        result = brain.think(user_input, identity="sylana", active_tools=["outreach", "trading", "analytics"])
        print(f"Sylana: {result['response']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
