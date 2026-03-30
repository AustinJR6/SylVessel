# dashboard/views/trade_history_view.py

import streamlit as st
import pandas as pd
from typing import List, Dict


def show_trade_history(trades: List[Dict]):
    """Render a table of recent trades."""
    st.header("📜 Trade History")
    if not trades:
        st.info("No trades recorded yet.")
        return
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    st.dataframe(df)
