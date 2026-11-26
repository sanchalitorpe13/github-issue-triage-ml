"""
Reusable UI Components & Result Visualizers.
Renders clean HTML/Streamlit cards for Classification, Priority, Severity, and Action Items.
"""
from typing import Any, Dict, List, Optional
import streamlit as st


def render_category_badge(category: str) -> str:
    """Returns HTML for category badge."""
    c = category.lower().strip()
    badge_class = "badge-bug"
    if "feature" in c:
        badge_class = "badge-feature"
    elif "doc" in c:
        badge_class = "badge-docs"
    elif "question" in c or "support" in c:
        badge_class = "badge-question"
    return f'<span class="badge {badge_class}">{category}</span>'


def render_priority_badge(level: str, name: str = "") -> str:
    """Returns HTML for priority badge."""
    lvl = level.upper().strip()
    badge_class = f"badge-{lvl.lower()}" if lvl.lower() in ["p0", "p1", "p2", "p3", "p4"] else "badge-p2"
    label_text = f"{lvl} - {name}" if name else lvl
    return f'<span class="badge {badge_class}">{label_text}</span>'


def render_status_pill(is_online: bool, text: str = "") -> None:
    """Renders connection status indicator."""
    if is_online:
        html = f'<span class="status-pill status-online">● {text or "Backend Connected"}</span>'
    else:
        html = f'<span class="status-pill status-offline">● {text or "Backend Disconnected"}</span>'
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(title: str, description: str, icon: str = "📊") -> None:
    """Renders a styled empty state card."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div style="font-size: 2.2rem; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #E2E8F0; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 0.9rem; color: #94A3B8; max-width: 480px; margin: 0 auto;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_triage_result_view(result: Dict[str, Any]) -> None:
    """
    Renders complete, interactive, professional visualization of a TriageResult.
    """
    cat = result.get("predicted_category", "Unknown")
    conf = float(result.get("confidence", 0.0))
    conf_status = result.get("confidence_status", "Medium")
    p_data = result.get("priority", {})
    s_data = result.get("severity", {})
    u_data = result.get("urgency", {})
    recs = result.get("recommendations", [])
    explanation = result.get("explanation", {})

    # Top Header Summary Card
    st.markdown(
        f"""
        <div class="saas-card" style="border-left: 4px solid #38BDF8;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 500;">ISSUE #{result.get("issue_id", "ANALYZED")}</span>
                    <h3 style="margin: 4px 0 0 0; font-size: 1.25rem; color: #F8FAFC;">{result.get("title")}</h3>
                </div>
                <div style="display: flex; gap: 8px;">
                    {render_category_badge(cat)}
                    {render_priority_badge(p_data.get("level", "P2"), p_data.get("name", ""))}
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Target SLA: <strong style="color: #38BDF8;">{p_data.get("sla_hours", 24)} hours</strong> &nbsp;|&nbsp; 
                Model: <code>{result.get("model_name", "Complement Naive Bayes")}</code> (v{result.get("model_version", "2.0.0")})
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4-Column Metric Grid
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="saas-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Category</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{cat}</div>
                <div style="font-size: 0.8rem; color: #38BDF8;">Confidence: {conf * 100:.1f}% ({conf_status})</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        score = p_data.get("score", 25)
        p_lvl = p_data.get("level", "P2")
        st.markdown(
            f"""
            <div class="saas-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Priority Tier</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{p_lvl} ({p_data.get('name', '')})</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Score: {score}/100 (Heuristic)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        sev_lvl = s_data.get("level", "Medium")
        st.markdown(
            f"""
            <div class="saas-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Technical Severity</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{sev_lvl}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Avail: {s_data.get('factors', {}).get('availability', 'Normal')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        urg_lvl = u_data.get("level", "Medium")
        st.markdown(
            f"""
            <div class="saas-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Operational Urgency</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{urg_lvl}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Signals: {len(u_data.get('signals', []))} detected</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detailed Cards Layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Priority Breakdown Card
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">
                    🎯 Priority Assessment Breakdown (Method: {p_data.get('method', 'rule_based')})
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 12px;">
                    <em>Note: Priority scores are rule-based heuristic assessments for triage guidance, not ML ground truth.</em>
                </div>
                <ul style="margin: 0; padding-left: 18px; font-size: 0.85rem; color: #94A3B8;">
            """,
            unsafe_allow_html=True
        )
        for r in p_data.get("reasoning", []):
            st.markdown(f"- {r}")

        if p_data.get("signals_detected"):
            st.markdown("**Detected Triggers:**")
            for s in p_data.get("signals_detected"):
                st.markdown(f"- 🚩 `{s}`")
        st.markdown("</div>", unsafe_allow_html=True)

        # Alternative Predictions
        alts = result.get("alternative_predictions", [])
        if alts:
            st.markdown(
                """
                <div class="saas-card">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">
                        📊 Probability Distribution Across Classes
                    </div>
                """,
                unsafe_allow_html=True
            )
            for a in alts:
                st.markdown(f"**{a.get('category')}**: `{a.get('percentage')}` ({a.get('probability')})")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Action Recommendations Card
        st.markdown(
            """
            <div class="saas-card">
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">
                    🛠️ Recommended Maintainer Actions
                </div>
            """,
            unsafe_allow_html=True
        )
        if recs:
            for rec in recs:
                st.markdown(f"- {rec}")
        else:
            st.markdown("- *Standard triage: assign to component maintainer.*")
        st.markdown("</div>", unsafe_allow_html=True)

        # Explainability & Influential Keywords
        top_kws = explanation.get("top_keywords", [])
        st.markdown(
            """
            <div class="saas-card">
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">
                    🔍 Decision Explainability & Influential Tokens
                </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f"**Rationale:** {explanation.get('category_reasoning', '')}")
        if top_kws:
            st.markdown("**Top TF-IDF N-grams:**")
            kw_tags = " ".join([f"`{kw}` ({score:.3f})" for kw, score in top_kws[:5]])
            st.markdown(kw_tags)
        st.markdown("</div>", unsafe_allow_html=True)

    # Raw Result Accordion
    with st.expander("📄 View Full JSON Schema Payload"):
        st.json(result)
