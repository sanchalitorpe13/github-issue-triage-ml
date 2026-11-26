"""
Streamlit Application Entrypoint.
Orchestrates midnight theme styling, sidebar navigation, backend connectivity monitoring,
and modular page routing for the Issue Intelligence Platform.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.state import init_session_state, get_api_client
from app.ui.theme import apply_midnight_theme
from app.ui.components import render_status_pill
from app.pages_ui.overview import render_overview_page
from app.pages_ui.analyzer import render_analyzer_page
from app.pages_ui.analytics import render_analytics_page
from app.pages_ui.model_info import render_model_info_page
from app.pages_ui.settings_page import render_settings_page

# Page Configuration
st.set_page_config(
    page_title="Issue Intelligence | AI Triage",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State & Midnight CSS
init_session_state()
apply_midnight_theme()

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">🤖 Issue Intelligence</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">AI-Powered GitHub Triage ML 2.0</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "🔍 Issue Analyzer",
            "📈 Session Analytics",
            "🧠 Model Information",
            "⚙️ Settings & API"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #1E293B; margin: 20px 0;'>", unsafe_allow_html=True)

    # Sidebar Connection Status Indicator
    client = get_api_client()
    try:
        health = client.check_health()
        is_up = health.get("status") in ["healthy", "operational"]
    except Exception:
        is_up = False

    render_status_pill(is_up, "Backend Online" if is_up else "Backend Offline")
    st.caption(f"Host: `{client.base_url}`")

# Route to Selected Page
if "Overview" in page:
    render_overview_page()
elif "Analyzer" in page:
    render_analyzer_page()
elif "Analytics" in page:
    render_analytics_page()
elif "Model" in page:
    render_model_info_page()
elif "Settings" in page:
    render_settings_page()
