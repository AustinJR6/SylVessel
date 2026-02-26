from __future__ import annotations

from core.brain import Brain
from core.logging_config import configure_logging


def main() -> int:
    configure_logging()
    brain = Brain.create_default(mode="quantized")

    print("SylVessel Quantized CLI")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        result = brain.think(user_input, identity="sylana", active_tools=[])
        print(f"Sylana: {result['response']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
