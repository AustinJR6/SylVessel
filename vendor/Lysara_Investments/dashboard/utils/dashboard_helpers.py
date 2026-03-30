# dashboard/utils/dashboard_helpers.py

import json
from pathlib import Path

def load_control_flags():
    """
    Load control flags from the dashboard's control JSON.
    These can be used by the bot launcher or services to respond to user commands.
    """
    flag_file = Path("dashboard/controls/control_flags.json")
    if flag_file.exists():
        try:
            return json.loads(flag_file.read_text())
        except json.JSONDecodeError:
            return {}
    return {}

import time
import streamlit as st

def auto_refresh(interval: int = 5):
    """Simple auto-refresh using session state."""
    if "_last_refresh" not in st.session_state:
        st.session_state["_last_refresh"] = time.time()
    elif time.time() - st.session_state["_last_refresh"] > interval:
        st.session_state["_last_refresh"] = time.time()
        st.rerun()

