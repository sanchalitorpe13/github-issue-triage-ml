"""
Settings & Backend Configuration Page.
Allows configuring the FastAPI backend URL, testing connection, and resetting session history.
"""
import streamlit as st
from app.state import clear_session_history, get_api_client
from app.ui.components import render_status_pill


def render_settings_page() -> None:
    """Renders connection settings and session control panel."""
    st.markdown('<div class="brand-title">⚙️ Settings & System Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem;'>"
        "Configure REST API connectivity, connection timeouts, and session data management."
        "</p>",
        unsafe_allow_html=True
    )

    client = get_api_client()

    with st.form("settings_form"):
        st.markdown("<h4 style='color: #F8FAFC; font-size: 1.05rem; margin-bottom: 12px;'>FastAPI Backend Connectivity</h4>", unsafe_allow_html=True)

        current_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
        new_url = st.text_input(
            "API Base URL",
            value=current_url,
            help="Base URL for the FastAPI backend service (default: http://127.0.0.1:8000)"
        )

        current_timeout = st.session_state.get("api_timeout", 15)
        new_timeout = st.slider(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=60,
            value=current_timeout,
            step=5
        )

        col_save, col_test = st.columns([1, 1])
        with col_save:
            save_clicked = st.form_submit_button("💾 Save Settings", use_container_width=True)

    if save_clicked:
        st.session_state["api_url"] = new_url.rstrip("/")
        st.session_state["api_timeout"] = new_timeout
        st.success("Settings updated successfully!")

    # Test Connection Section
    st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #F8FAFC; font-size: 1.05rem;'>Connection Diagnostic Test</h4>", unsafe_allow_html=True)

    if st.button("🔌 Test Backend Connection", use_container_width=True):
        test_client = get_api_client()
        with st.spinner(f"Pinging backend at {test_client.base_url}..."):
            try:
                health = test_client.check_health()
                st.success(f"Connection successful! Backend is online and operational at {test_client.base_url}")
                st.json(health)
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

    # Clear History Section
    st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #F8FAFC; font-size: 1.05rem;'>Session Data Management</h4>", unsafe_allow_html=True)

    history_len = len(st.session_state.get("session_history", []))
    st.markdown(f"Current session holds **{history_len}** analyzed issue records in browser memory.")

    if st.button("🗑️ Clear Session Analysis History", use_container_width=True):
        clear_session_history()
        st.success("Session history cleared successfully!")
