import streamlit as st
import pandas as pd
import plotly.express as px


def show_equity_curve(data: list[dict]):
    """Render the equity curve line chart."""
    st.header("📈 Equity Curve")
    if not data:
        st.info("No equity data available.")
        return
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig = px.line(df, x="timestamp", y="equity")
    st.plotly_chart(fig, use_container_width=True)
