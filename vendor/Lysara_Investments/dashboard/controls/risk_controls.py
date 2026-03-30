# dashboard/controls/risk_controls.py

import streamlit as st
import json
from pathlib import Path

SETTINGS_FILE = Path("config") / "settings_crypto.json"

def _save_settings(new_vals: dict):
    """Write updated risk settings back to the JSON file."""
    try:
        cfg = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        cfg = {}
    cfg.update(new_vals)
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=2))

def show_risk_controls():
    st.sidebar.header("⚖️ Risk Controls")

    # Load existing defaults
    try:
        current = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        current = {}

    max_dd = st.sidebar.slider(
        "Max Drawdown (%)", min_value=0.0, max_value=1.0,
        value=current.get("max_drawdown", 0.20), step=0.01
    )
    risk_per = st.sidebar.slider(
        "Risk per Trade (%)", min_value=0.0, max_value=0.1,
        value=current.get("risk_per_trade", 0.02), step=0.005
    )

    if st.sidebar.button("Update Risk Settings"):
        _save_settings({
            "max_drawdown": max_dd,
            "risk_per_trade": risk_per
        })
        st.sidebar.success("✅ Risk settings updated")
