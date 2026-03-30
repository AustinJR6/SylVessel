# dashboard/views/log_view.py

import streamlit as st
from typing import List


def show_log_view(lines: List[str]):
    """Display scrollable log output."""
    st.header("📝 Log Output")
    if not lines:
        st.info("Log file is empty.")
        return
    st.text_area("Logs", value="\n".join(lines), height=300)
