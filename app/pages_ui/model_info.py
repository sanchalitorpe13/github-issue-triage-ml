"""
Model Information & Architecture Page.
Fetches and displays live machine learning model metadata, benchmark evaluation scores,
and TF-IDF vectorizer configuration retrieved directly from the FastAPI backend.
"""
import streamlit as st
from app.state import get_api_client
from app.ui.components import render_empty_state


def render_model_info_page() -> None:
    """Renders active ML model architecture and evaluation benchmark reports."""
    st.markdown('<div class="brand-title">🧠 Machine Learning Model Architecture & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.95rem;'>"
        "Active classification pipeline parameters and verified benchmark evaluation metrics retrieved from the backend."
        "</p>",
        unsafe_allow_html=True
    )

    client = get_api_client()

    try:
        data = client.get_model_info()
    except Exception as e:
        render_empty_state(
            title="Backend Model Info Unavailable",
            description=f"Cannot retrieve model metadata: {str(e)}. Please check if the FastAPI backend is running.",
            icon="⚠️"
        )
        return

    # Model Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Algorithm</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">{data.get('model_name', 'ComplementNB')}</div>
                <div style="font-size: 0.8rem; color: #38BDF8;">Imbalanced Text Specialist</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Pipeline Version</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin: 4px 0;">v{data.get('version', '2.0.0')}</div>
                <div style="font-size: 0.8rem; color: #34D399;">Production Deployed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        metrics = data.get("metrics_summary", {})
        macro_f1 = metrics.get("macro_f1", 0.0)
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Selection Metric (Macro F1)</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #818CF8; margin: 4px 0;">{macro_f1:.4f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Target Metric</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        acc = metrics.get("accuracy", 0.0)
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Test Set Accuracy</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #38BDF8; margin: 4px 0;">{acc*100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Held-Out Test Split</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detailed Architecture Specs
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 12px;">
                    🎯 Target Classification Taxonomy
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    The model was trained on multi-repository open-source developer issues mapped to 4 canonical categories:
                </p>
                <ul style="font-size: 0.85rem; color: #CBD5E1; padding-left: 18px;">
                    <li><strong style="color: #F87171;">Bug:</strong> Software crashes, segmentation faults, regressions, functional errors</li>
                    <li><strong style="color: #60A5FA;">Feature:</strong> Enhancements, feature proposals, API additions</li>
                    <li><strong style="color: #34D399;">Documentation:</strong> Guide updates, broken links, typos, tutorials</li>
                    <li><strong style="color: #FBBF24;">Question:</strong> Troubleshooting, configuration queries, community support</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_r:
        vec = data.get("vectorizer_summary", {})
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 12px;">
                    ⚙️ TF-IDF Feature Extractor Configuration
                </div>
                <ul style="font-size: 0.85rem; color: #CBD5E1; padding-left: 18px;">
                    <li><strong>N-Gram Range:</strong> {vec.get('ngram_range', [1, 2])} (Unigrams + Bigrams)</li>
                    <li><strong>Max Features:</strong> {vec.get('max_features', 5000):,} terms</li>
                    <li><strong>Learned Vocabulary Size:</strong> {vec.get('vocab_size', 1929):,} features</li>
                    <li><strong>Sublinear TF Scaling:</strong> Enabled (1 + log(tf))</li>
                    <li><strong>Token Pattern:</strong> Preserves code identifiers and hyphens</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Methodology & Disclaimer Card
    st.markdown(
        """
        <div class="saas-card" style="border-left: 4px solid #818CF8;">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 6px;">
                💡 Engineering Methodology Note
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                • <strong>Issue Type:</strong> Supervised machine learning prediction using TF-IDF n-grams with Complement Naive Bayes.<br>
                • <strong>Priority (P0–P4):</strong> Deterministic rule-based heuristic scoring based on crash severity, security tags, data loss risk, and workaround mitigations.<br>
                • <strong>Severity & Urgency:</strong> Orthogonal multidimensional evaluations assessing blast radius and operational time sensitivity.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
