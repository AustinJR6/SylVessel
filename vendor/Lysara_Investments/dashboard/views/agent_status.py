import json
from pathlib import Path
import streamlit as st

LOG_PATH = "logs/agent_history.json"


def _load_last_entry():
    path = Path(LOG_PATH)
    if not path.is_file():
        return None
    try:
        line = path.read_text().strip().splitlines()[-1]
        return json.loads(line)
    except Exception:
        return None


def show_agent_status(info=None, auto_mode=True):
    """Display the most recent agent status on the dashboard.

    Parameters
    ----------
    info : dict, optional
        Dictionary containing agent decision information. If ``None`` the
        function will attempt to load the last entry from
        ``LOG_PATH`` for backward compatibility.
    auto_mode : bool, optional
        Current autonomous mode flag used to set the checkbox state.
    """

    st.header("🤖 Agent Status")
    st.checkbox("Autonomous Mode", value=auto_mode, key="autonomous")

    if info is None:
        info = _load_last_entry()

    if not info:
        st.info("No agent activity yet.")
        return

    st.subheader(f"Last decision for {info.get('ticker', 'N/A')}")
    st.write(f"Price: {info.get('price')}")
    decision = info.get('decision', {})
    st.write(f"Decision: {decision.get('action')}")
    st.write(f"Confidence: {decision.get('confidence')}")
    st.write(f"Rationale: {decision.get('rationale')}")
    st.write(decision.get('explanation'))

    if st.session_state.get("pending_trade"):
        col1, col2 = st.columns(2)
        if col1.button("Approve Trade"):
            st.session_state["pending_trade"] = False
        if col2.button("Reject Trade"):
            st.session_state["pending_trade"] = False
