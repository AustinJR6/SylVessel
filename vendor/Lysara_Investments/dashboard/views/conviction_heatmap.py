import streamlit as st
import pandas as pd


def show_conviction_heatmap(sentiment: dict):
    """Render a simple heatmap of sentiment scores per asset."""
    st.header("🔥 Conviction Heatmap")
    cp = sentiment.get("cryptopanic", {}) if sentiment else {}
    if not cp:
        st.info("No sentiment data available.")
        return
    st.caption("Source: CryptoPanic sentiment")
    data = {
        "asset": list(cp.keys()),
        "score": [v.get("score", 0.0) for v in cp.values()],
    }
    df = pd.DataFrame(data).set_index("asset")
    styled = df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1)
    st.dataframe(styled, height=300)
