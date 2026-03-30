import pandas as pd
import streamlit as st


def show_forex_view(data, timeframe: str = "1h"):
    """Display forex market chart with optional SMA and RSI overlays."""
    st.header("Forex Market Overview")
    st.caption(f"Timeframe: {timeframe}")

    if not data:
        st.write("No forex market data available.")
        return

    try:
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time")
        if "sma" in df.columns:
            st.line_chart(df[["price", "sma"]])
        else:
            st.line_chart(df["price"])
        if "rsi" in df.columns:
            st.line_chart(df["rsi"])
    except Exception as exc:
        st.error(f"Failed to render forex chart: {exc}")
