"""
Overview Dashboard Page.
Displays backend connection status, model readiness, real session metrics, and recent analyses.
"""
from typing import Any, Dict, List
import streamlit as st
from app.state import get_api_client
from app.ui.components import render_empty_state, render_status_pill, render_category_badge, render_priority_badge


def render_overview_page() -> None:
    """Renders the top-level overview and session summary."""
    st.markdown('<div class="brand-title">Issue Intelligence & Triage ML</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem; margin-top: 4px;'>"
        "Production AI Platform for automated GitHub issue classification, blast-radius severity evaluation, and SLA prioritization."
        "</p>",
        unsafe_allow_html=True
    )

    client = get_api_client()
    is_online = False
    health_data = {}

    try:
        health_data = client.check_health()
        is_online = health_data.get("status") in ["healthy", "operational"]
    except Exception as e:
        is_online = False

    # Top Status Bar Card
    col_status, col_meta = st.columns([1, 2])
    with col_status:
        render_status_pill(is_online, "FastAPI Backend Online" if is_online else "Backend Offline")

    with col_meta:
        if is_online:
            st.markdown(
                f"<div style='font-size: 0.85rem; color: #94A3B8; text-align: right;'>"
                f"Model: <strong style='color: #E2E8F0;'>{health_data.get('supported_classes', ['Bug', 'Feature', 'Docs', 'Question'])}</strong> &nbsp;|&nbsp; "
                f"Engine: <span style='color: #34D399;'>{health_data.get('triage_engine_status', 'Operational')}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size: 0.85rem; color: #F87171; text-align: right;'>"
                f"Cannot reach <code>{client.base_url}</code>. Start FastAPI server to analyze issues."
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<hr style='border-color: #1E293B; margin: 16px 0 24px 0;'>", unsafe_allow_html=True)

    # Session Metrics Overview
    history: List[Dict[str, Any]] = st.session_state.get("session_history", [])
    total_analyzed = len(history)

    if total_analyzed > 0:
        bug_count = sum(1 for h in history if h.get("predicted_category") == "Bug")
        critical_count = sum(1 for h in history if h.get("priority", {}).get("level") in ["P0", "P1"])
        avg_conf = sum(h.get("confidence", 0.0) for h in history) / total_analyzed

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="saas-card">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Issues Analyzed</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">{total_analyzed}</div>
                    <div style="font-size: 0.8rem; color: #38BDF8;">Active Session Data</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="saas-card">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Bugs Identified</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #F87171;">{bug_count}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">{(bug_count/total_analyzed)*100:.0f}% of session total</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="saas-card">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">High / Critical (P0-P1)</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #FB923C;">{critical_count}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">{(critical_count/total_analyzed)*100:.0f}% SLA Urgent</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="saas-card">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Mean Confidence</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #34D399;">{avg_conf*100:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Calibrated Model Score</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Recent Session History
        st.markdown("<h4 style='color: #F8FAFC; margin-top: 24px; font-size: 1.1rem;'>Recent Analyzed Issues</h4>", unsafe_allow_html=True)
        for item in history[:5]:
            p_lvl = item.get("priority", {}).get("level", "P2")
            p_name = item.get("priority", {}).get("name", "")
            cat = item.get("predicted_category", "Unknown")
            st.markdown(
                f"""
                <div class="saas-card" style="padding: 14px 18px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 0.75rem; color: #94A3B8;">#{item.get('issue_id', '-')}</span>
                            <strong style="color: #F8FAFC; margin-left: 8px;">{item.get('title')}</strong>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            {render_category_badge(cat)}
                            {render_priority_badge(p_lvl, p_name)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        render_empty_state(
            title="No Issues Analyzed in Current Session",
            description="Submit your first GitHub issue in the Issue Analyzer to populate live session analytics and triage insights.",
            icon="⚡"
        )
