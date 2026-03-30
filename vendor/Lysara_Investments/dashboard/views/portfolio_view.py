import streamlit as st
import pandas as pd
from typing import List, Dict


def show_portfolio_table(positions: List[Dict], title: str):
    """Render a simple table of portfolio holdings."""
    st.subheader(title)
    if not positions:
        st.info("No data available.")
        return
    df = pd.DataFrame(positions)
    st.dataframe(df)


def show_sim_summary(summary: Dict, balance: float):
    st.metric("Sim Balance", f"${balance:,.2f}")
    if summary:
        st.write(
            f"Win Rate: {summary.get('win_rate', 0)}% | Avg Return: {summary.get('avg_return',0)} | Trades: {summary.get('trade_count',0)}"
        )
