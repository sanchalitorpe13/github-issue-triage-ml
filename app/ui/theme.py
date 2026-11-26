"""
Midnight SaaS Theme & Custom CSS Styles.
Implements a developer-grade dark navy theme with subtle cyan and violet accents.
"""
import streamlit as st


def apply_midnight_theme() -> None:
    """Injects custom midnight CSS into the Streamlit application."""
    css = """
    <style>
    /* Global Application Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #0A0E1A;
        color: #E2E8F0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #070B14;
        border-right: 1px solid #1E293B;
    }

    /* Card Containers */
    .saas-card {
        background: #111726;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
    }

    .saas-card:hover {
        border-color: #334155;
    }

    /* Header Accent */
    .brand-title {
        font-weight: 700;
        font-size: 1.5rem;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    /* Metric Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-bug { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-feature { background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-docs { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-question { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .badge-p0 { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    .badge-p1 { background: rgba(249, 115, 22, 0.2); color: #FB923C; border: 1px solid #FB923C; }
    .badge-p2 { background: rgba(59, 130, 246, 0.2); color: #38BDF8; border: 1px solid #38BDF8; }
    .badge-p3 { background: rgba(16, 185, 129, 0.2); color: #4ADE80; border: 1px solid #4ADE80; }
    .badge-p4 { background: rgba(148, 163, 184, 0.15); color: #94A3B8; border: 1px solid #475569; }

    /* Connection Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .status-online { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .status-offline { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }

    /* Modern Progress / Bar */
    .bar-bg {
        background-color: #1E293B;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        margin-top: 4px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #38BDF8;
        color: #38BDF8;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%);
        border: none;
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 15px rgba(2, 132, 199, 0.4);
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        background: #0E131F;
        border: 1px dashed #1E293B;
        border-radius: 12px;
        color: #64748B;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
