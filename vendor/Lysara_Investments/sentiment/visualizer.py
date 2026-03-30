"""Streamlit dashboards for sentiment visualization."""
from __future__ import annotations

import streamlit as st
import pandas as pd


def show_heatmap(data: pd.DataFrame) -> None:
    """Display a simple heatmap of sentiment scores."""
    st.title("Sentiment Heatmap")
    st.dataframe(data)


# TODO: Add more detailed visualizations
