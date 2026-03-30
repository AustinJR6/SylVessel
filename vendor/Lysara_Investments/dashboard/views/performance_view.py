# dashboard/views/performance_view.py

import streamlit as st
from typing import Dict


def show_performance_view(metrics: Dict):
    """Display win rate and average return statistics."""
    st.header("📊 Performance Metrics")
    if not metrics:
        st.info("No performance data available.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{metrics.get('win_rate', 0)}%")
    col2.metric("Avg Return", f"{metrics.get('avg_return', 0)}")
    col3.metric("Open Risk", metrics.get('open_risk', 0))
