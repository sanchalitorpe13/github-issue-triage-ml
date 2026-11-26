"""
Chart Builders and Renderers for Session Analytics.
Supports Altair natively (built-in, dark-theme optimized) with optional Plotly fallback.
Renders responsive donut charts, priority bars, and confidence distributions.
"""
from typing import Any, Dict, List
import pandas as pd
import altair as alt
import streamlit as st

# Check optional Plotly availability
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Theme palette matching Midnight Theme
CATEGORY_COLOR_MAP = {
    "Bug": "#F87171",
    "Feature": "#60A5FA",
    "Documentation": "#34D399",
    "Question": "#FBBF24"
}

PRIORITY_COLOR_MAP = {
    "P0": "#EF4444",
    "P1": "#FB923C",
    "P2": "#38BDF8",
    "P3": "#4ADE80",
    "P4": "#94A3B8"
}


def _apply_dark_theme(chart: alt.Chart) -> alt.Chart:
    """Applies midnight dark theme styling to an Altair chart."""
    return chart.configure(
        background="transparent"
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        fontSize=14,
        font="Inter, system-ui, sans-serif",
        color="#F8FAFC",
        anchor="start"
    ).configure_axis(
        labelColor="#94A3B8",
        titleColor="#E2E8F0",
        gridColor="#1E293B",
        domainColor="#334155",
        tickColor="#334155",
        labelFont="Inter, system-ui, sans-serif",
        titleFont="Inter, system-ui, sans-serif"
    ).configure_legend(
        labelColor="#E2E8F0",
        titleColor="#94A3B8",
        labelFont="Inter, system-ui, sans-serif",
        titleFont="Inter, system-ui, sans-serif"
    )


def create_category_donut_chart(history: List[Dict[str, Any]]):
    """
    Creates donut chart showing category distribution.
    Returns an Altair Chart (or Plotly Figure if Plotly is available).
    """
    categories = [item.get("predicted_category", "Unknown") for item in history]
    df = pd.Series(categories).value_counts().reset_index()
    df.columns = ["Category", "Count"]

    cats = list(CATEGORY_COLOR_MAP.keys())
    colors = [CATEGORY_COLOR_MAP[c] for c in cats]

    base = alt.Chart(df).mark_arc(innerRadius=45, outerRadius=90, stroke="#0F172A", strokeWidth=2).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(
            field="Category",
            type="nominal",
            scale=alt.Scale(domain=cats, range=colors),
            legend=alt.Legend(title="Category", orient="bottom")
        ),
        tooltip=["Category:N", "Count:Q"]
    ).properties(
        title="Session Category Breakdown",
        height=260
    )

    return _apply_dark_theme(base)


def create_priority_bar_chart(history: List[Dict[str, Any]]):
    """
    Creates bar chart of priority tiers across session.
    Returns an Altair Chart.
    """
    tiers = [item.get("priority", {}).get("level", "P2") for item in history]
    order = ["P0", "P1", "P2", "P3", "P4"]
    counts = {k: 0 for k in order}
    for t in tiers:
        if t in counts:
            counts[t] += 1

    df = pd.DataFrame({"Priority": list(counts.keys()), "Count": list(counts.values())})
    colors = [PRIORITY_COLOR_MAP[k] for k in order]

    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Priority:N", sort=order, axis=alt.Axis(title="Priority Level", labelAngle=0)),
        y=alt.Y("Count:Q", axis=alt.Axis(title="Issue Count", tickMinStep=1)),
        color=alt.Color("Priority:N", scale=alt.Scale(domain=order, range=colors), legend=None),
        tooltip=["Priority:N", "Count:Q"]
    ).properties(
        title="Session Priority Breakdown",
        height=260
    )

    text = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        color="#F8FAFC",
        fontSize=12
    ).encode(
        text="Count:Q"
    )

    chart = bars + text
    return _apply_dark_theme(chart)


def create_confidence_distribution_chart(history: List[Dict[str, Any]]):
    """
    Creates histogram/binned bar chart of model confidence scores.
    Returns an Altair Chart.
    """
    scores = [float(item.get("confidence", 0.0)) for item in history]
    df = pd.DataFrame({"Confidence": scores})

    chart = alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
        color="#818CF8"
    ).encode(
        x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=10, extent=[0.0, 1.0]), axis=alt.Axis(title="Model Confidence (0.0 - 1.0)", format=".1f")),
        y=alt.Y("count()", axis=alt.Axis(title="Issue Count", tickMinStep=1)),
        tooltip=[alt.Tooltip("Confidence:Q", bin=alt.Bin(maxbins=10), title="Range"), alt.Tooltip("count()", title="Count")]
    ).properties(
        title="Confidence Score Distribution",
        height=220
    )

    return _apply_dark_theme(chart)


def render_category_chart(history: List[Dict[str, Any]]) -> None:
    """Renders category donut chart in Streamlit."""
    chart = create_category_donut_chart(history)
    st.altair_chart(chart, use_container_width=True)


def render_priority_chart(history: List[Dict[str, Any]]) -> None:
    """Renders priority bar chart in Streamlit."""
    chart = create_priority_bar_chart(history)
    st.altair_chart(chart, use_container_width=True)


def render_confidence_chart(history: List[Dict[str, Any]]) -> None:
    """Renders confidence distribution chart in Streamlit."""
    chart = create_confidence_distribution_chart(history)
    st.altair_chart(chart, use_container_width=True)
