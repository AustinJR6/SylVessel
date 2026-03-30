# dashboard/controls/trading_controls.py

import streamlit as st
import json
from pathlib import Path
import subprocess
import sys

CONTROL_FILE = Path("dashboard") / "controls" / "control_flags.json"


def _write_flags(flags: dict):
    """Persist control flags (start/stop commands) for bots to pick up."""
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)  # ✅ Ensures folder exists
    data = CONTROL_FILE.exists() and json.loads(CONTROL_FILE.read_text()) or {}
    data.update(flags)
    CONTROL_FILE.write_text(json.dumps(data, indent=2))


def show_trading_controls(sim_portfolio=None):
    st.sidebar.header("🔧 Trading Controls")

    if st.sidebar.button("Start Crypto Bot"):
        _write_flags({"start_crypto": True})
        st.sidebar.success("🚀 Crypto bot STARTED")

    if st.sidebar.button("Stop Crypto Bot"):
        _write_flags({"stop_crypto": True})
        st.sidebar.warning("⏸ Crypto bot STOPPED")

    if st.sidebar.button("Start All Bots"):
        _write_flags({"start_all": True})
        st.sidebar.success("🚀 All bots STARTED")

    if st.sidebar.button("Stop All Bots"):
        _write_flags({"stop_all": True})
        st.sidebar.warning("⏸ All bots STOPPED")

    if sim_portfolio and st.sidebar.button("Reset Simulation"):
        sim_portfolio.reset()
        st.sidebar.success("🧹 Simulation reset")

    if st.sidebar.button("Launch Onchain Agent"):
        subprocess.Popen([sys.executable, "chatbot.py"])  # noqa: S603
        st.sidebar.info("🤖 Onchain agent launched")
