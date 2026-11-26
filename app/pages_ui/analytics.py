"""
Session Analytics Page.
Visualizes category distributions, priority breakdowns, and confidence histograms
using interactive Altair charts based strictly on actual session data.
"""
from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from app.ui.charts import (
    render_category_chart,
    render_priority_chart,
    render_confidence_chart,
)
from app.ui.components import render_empty_state, render_category_badge, render_priority_badge


def render_analytics_page() -> None:
    """Renders interactive session analytics and historical breakdown."""
    st.markdown('<div class="brand-title">📈 Session Analytics & Triage Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem;'>"
        "Real-time analytics and distributions generated from issues analyzed during your active session."
        "</p>",
        unsafe_allow_html=True
    )

    history: List[Dict[str, Any]] = st.session_state.get("session_history", [])

    if not history:
        render_empty_state(
            title="No Session Analytics Available",
            description="Analytics update automatically as issues are processed. Analyze at least one issue in the Issue Analyzer to view interactive charts.",
            icon="📊"
        )
        return

    st.markdown(
        f"<div style='font-size: 0.85rem; color: #38BDF8; margin-bottom: 16px;'>"
        f"Displaying metrics for <strong>{len(history)}</strong> issues analyzed in this session."
        f"</div>",
        unsafe_allow_html=True
    )

    # 2-Column Chart Row
    c1, c2 = st.columns(2)
    with c1:
        render_category_chart(history)
    with c2:
        render_priority_chart(history)

    # Confidence Distribution Chart
    render_confidence_chart(history)

    # Session Table
    st.markdown("<h4 style='color: #F8FAFC; margin-top: 24px;'>Session Issue Ledger</h4>", unsafe_allow_html=True)

    table_data = []
    for item in history:
        table_data.append({
            "ID": item.get("issue_id", "-"),
            "Title": item.get("title", ""),
            "Category": item.get("predicted_category", ""),
            "Confidence": f"{float(item.get('confidence', 0.0)) * 100:.1f}%",
            "Priority": f"{item.get('priority', {}).get('level', '')} ({item.get('priority', {}).get('score', '')} pts)",
            "Severity": item.get("severity", {}).get("level", ""),
            "Urgency": item.get("urgency", {}).get("level", ""),
            "SLA": f"{item.get('priority', {}).get('sla_hours', '')}h"
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
