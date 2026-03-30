import argparse
import subprocess
import sys
import threading
import time
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def stream_output(proc: subprocess.Popen, label: str, buffer: List[str]):
    """Forward process output to the terminal while storing it."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            buffer.append(line)
            logging.info(f"[{label}] {line.strip()}")
    except Exception as exc:
        logging.error(f"Error reading {label} output: {exc}")


def start_process(cmd: List[str], label: str):
    """Start a subprocess and stream its output."""
    buffer: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=stream_output, args=(proc, label, buffer), daemon=True
        )
        thread.start()
        logging.info(f"{label} launched successfully.")
        return proc, buffer
    except FileNotFoundError as exc:
        logging.error(f"Failed to start {label}: {exc}")
        return None, buffer


def main():
    if sys.prefix == sys.base_prefix:
        logging.warning("⚠️ Warning: Not running inside the virtual environment!")
    parser = argparse.ArgumentParser(description="Launch trading bot and dashboard")
    subparsers = parser.add_subparsers(dest="command")

    # Default options for launching the trading bot
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--simulate", action="store_true", help="Run bot in simulation mode"
    )
    mode_group.add_argument(
        "--live", action="store_true", help="Run bot in live trading mode"
    )

    # Onchain agent command
    agent_parser = subparsers.add_parser("launch-agent", help="Run the onchain agent")
    agent_parser.add_argument(
        "--test",
        action="store_true",
        help="Simulate agent actions without blockchain interaction",
    )

    args = parser.parse_args()

    if args.command == "launch-agent":
        agent_cmd = [sys.executable, "chatbot.py"]
        if args.test:
            agent_cmd.append("--test")
        subprocess.run(agent_cmd, check=False)
        return

    bot_cmd = [sys.executable, "main.py"]
    if args.simulate:
        bot_cmd.append("--simulate")
    elif args.live:
        bot_cmd.append("--live")

    bot_proc, bot_buffer = start_process(bot_cmd, "Bot")
    dash_proc, dash_buffer = start_process(
        ["streamlit", "run", "dashboard/app.py"], "Dashboard"
    )

    if dash_proc:
        logging.info("Dashboard running at http://localhost:8501")

    try:
        while True:
            time.sleep(1)
            if bot_proc and bot_proc.poll() is not None:
                err = "".join(bot_buffer[-10:]).strip()
                msg = (
                    f"[!] Bot crashed with error: {err}"
                    if err
                    else f"[!] Bot exited with code {bot_proc.returncode}"
                )
                logging.error(msg)
                break
            if dash_proc and dash_proc.poll() is not None:
                err = "".join(dash_buffer[-10:]).strip()
                msg = (
                    f"[!] Dashboard crashed with error: {err}"
                    if err
                    else f"[!] Dashboard exited with code {dash_proc.returncode}"
                )
                logging.error(msg)
                break
    except KeyboardInterrupt:
        logging.info("Interrupt received. Shutting down...")
    finally:
        for proc, label in [(bot_proc, "Bot"), (dash_proc, "Dashboard")]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                logging.info(f"{label} terminated.")


if __name__ == "__main__":
    main()
