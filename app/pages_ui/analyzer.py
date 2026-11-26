"""
Issue Analyzer Page.
Interactive input form with sample issue loader, validation, API dispatch, and complete triage visualization.
"""
import json
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st
from app.state import add_to_session_history, get_api_client
from app.ui.components import render_triage_result_view
from config.settings import settings


def load_curated_examples() -> List[Dict[str, Any]]:
    """Loads authentic test issues from sample_issues.json."""
    sample_file = settings.DATA_PATH / "sample_issues.json"
    if sample_file.exists():
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def render_analyzer_page() -> None:
    """Renders the interactive issue triage analyzer interface."""
    st.markdown('<div class="brand-title">🔍 AI Issue Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem;'>"
        "Submit a GitHub issue title and description to run ML category classification and intelligent priority scoring."
        "</p>",
        unsafe_allow_html=True
    )

    examples = load_curated_examples()
    example_titles = ["-- Select an authentic sample issue --"] + [f"[{ex.get('repository', '')}] {ex.get('title', '')}" for ex in examples]

    selected_example_idx = st.selectbox(
        "⚡ Quick Test: Load Curated Real-World Issue",
        range(len(example_titles)),
        format_func=lambda i: example_titles[i],
        help="Select a real open-source GitHub issue to automatically populate the input fields."
    )

    # Defaults
    default_title = ""
    default_body = ""
    default_repo = ""
    default_id = ""

    if selected_example_idx > 0:
        ex = examples[selected_example_idx - 1]
        default_title = ex.get("title", "")
        default_body = ex.get("body", "")
        default_repo = ex.get("repository", "")
        default_id = str(ex.get("id", ""))

    with st.form("triage_form"):
        title_input = st.text_input(
            "Issue Title *",
            value=default_title,
            placeholder="e.g. Fatal error: Segmentation fault in QuantizedConv2D",
            help="Required. 1-500 characters."
        )

        body_input = st.text_area(
            "Issue Description / Traceback",
            value=default_body,
            height=160,
            placeholder="Paste complete issue description, error logs, or stack traces...",
            help="Optional. Up to 50,000 characters."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            repo_input = st.text_input("Repository (optional)", value=default_repo, placeholder="owner/repo")
        with col2:
            num_input = st.text_input("Issue Number / ID (optional)", value=default_id, placeholder="e.g. 1024")
        with col3:
            labels_input = st.text_input("Existing Labels (optional)", placeholder="comma-separated, e.g. bug, core")

        submitted = st.form_submit_button("🚀 Analyze Issue with AI", use_container_width=True)

    if submitted:
        if not title_input.strip():
            st.error("Please provide an issue title. The title cannot be blank.")
            return

        client = get_api_client()

        with st.spinner("Analyzing text with NLP model and evaluating priority rules..."):
            try:
                # Parse labels
                parsed_labels = [l.strip() for l in labels_input.split(",") if l.strip()] if labels_input else []
                issue_num = int(num_input) if num_input.isdigit() else None

                result = client.triage_issue(
                    title=title_input,
                    body=body_input,
                    repository=repo_input if repo_input else None,
                    issue_number=issue_num,
                    labels=parsed_labels if parsed_labels else None
                )

                st.session_state["current_result"] = result
                add_to_session_history(result)
                st.success("Triage analysis complete!")

            except Exception as e:
                st.error(f"Triage execution failed: {str(e)}")

    # Display Current Result
    current = st.session_state.get("current_result")
    if current:
        st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'>", unsafe_allow_html=True)
        render_triage_result_view(current)
