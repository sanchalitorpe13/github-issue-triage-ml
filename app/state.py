"""
Streamlit Session State Manager.
Maintains session-level issue history, deduplication, and client configuration.
"""
import os
from typing import Any, Dict, List, Optional
import streamlit as st
from app.client import TriageAPIClient


def init_session_state() -> None:
    """Initializes Streamlit session state keys if not already set."""
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = os.getenv("TRIAGE_API_URL", "http://127.0.0.1:8000")

    if "api_timeout" not in st.session_state:
        st.session_state["api_timeout"] = 15

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = []

    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None

    if "last_analyzed_hash" not in st.session_state:
        st.session_state["last_analyzed_hash"] = None


def get_api_client() -> TriageAPIClient:
    """Returns a configured TriageAPIClient using session state parameters."""
    return TriageAPIClient(
        base_url=st.session_state.get("api_url", "http://127.0.0.1:8000"),
        timeout=st.session_state.get("api_timeout", 15)
    )


def add_to_session_history(triage_result: Dict[str, Any]) -> None:
    """
    Appends a successfully triaged issue to session history without duplicate entries.
    """
    history: List[Dict[str, Any]] = st.session_state.get("session_history", [])

    # Unique identifier based on title and analyzed_at
    title = triage_result.get("title", "")
    analyzed_at = triage_result.get("analyzed_at", "")
    item_hash = f"{title}_{analyzed_at}"

    if st.session_state.get("last_analyzed_hash") == item_hash:
        return

    # Check for existing matching entry
    if not any(h.get("title") == title and h.get("analyzed_at") == analyzed_at for h in history):
        history.insert(0, triage_result)
        st.session_state["session_history"] = history
        st.session_state["last_analyzed_hash"] = item_hash


def clear_session_history() -> None:
    """Clears all session-stored issue analyses."""
    st.session_state["session_history"] = []
    st.session_state["current_result"] = None
    st.session_state["last_analyzed_hash"] = None
