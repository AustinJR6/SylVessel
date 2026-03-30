import streamlit as st
import pandas as pd


def show_ai_thought_feed(entries: list[dict]):
    """Display recent AI strategist decisions."""
    st.header("🤖 AI Thought Feed")
    if not entries:
        st.info("No AI decisions logged yet.")
        return
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.dataframe(df)
