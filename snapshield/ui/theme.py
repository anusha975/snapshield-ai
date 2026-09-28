"""SnapShield AI - Design System and Theming.

Design Language Specification:
PRODUCT: SnapShield AI
TAGLINE: Your local privacy checkpoint for AI.

COLOR SYSTEM:
- Ink: #12181A (Primary text, deep dark mineral ink)
- Paper: #F2EEE2 (Primary surface background, warm textured paper)
- Verified Green: #2E7D5B (Status ok, verified hardware, allow actions)
- Redact Amber: #C97A2B (Redaction action, warnings, medium risk)
- Block Red: #B23A3A (Block action, critical risk, perimeter violations)
- Rule: #5A6B70 (Structural borders, secondary labels, divider lines)

TYPOGRAPHY:
- Space Grotesk: Interface text, titles, headings, controls, navigation
- IBM Plex Mono: Data, detection values, tags, performance telemetry, tokens, hashes

Architectural Representation:
RAW DOCUMENT -> CHECKPOINT -> PROTECTED OUTPUT
"""

from __future__ import annotations

# Exact specified color system tokens
COLOR_INK = "#12181A"
COLOR_PAPER = "#F2EEE2"
COLOR_VERIFIED_GREEN = "#2E7D5B"
COLOR_REDACT_AMBER = "#C97A2B"
COLOR_BLOCK_RED = "#B23A3A"
COLOR_RULE = "#5A6B70"

# Secondary surface tokens derived from Paper & Ink
COLOR_PAPER_CARD = "#FAF7EE"
COLOR_PAPER_MUTED = "#E8E2D4"
COLOR_INK_MUTED = "#3E4C50"

# Typography tokens
FONT_INTERFACE = "'Space Grotesk', -apple-system, sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SF Mono', Consolas, monospace"


def get_theme_css() -> str:
    """Returns the complete CSS stylesheet applying the SnapShield AI design language."""
    return f"""
    <style>
        /* Google Fonts: Space Grotesk (UI) + IBM Plex Mono (Data & Telemetry) */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        /* Base Body and Streamlit Application Canvas */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {COLOR_PAPER} !important;
            color: {COLOR_INK} !important;
            font-family: {FONT_INTERFACE} !important;
            -webkit-font-smoothing: antialiased;
        }}

        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        /* Typography Defaults */
        h1, h2, h3, h4, h5, h6 {{
            font-family: {FONT_INTERFACE} !important;
            color: {COLOR_INK} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }}

        p, span, label, div {{
            font-family: {FONT_INTERFACE};
            color: {COLOR_INK};
        }}

        /* Monospace elements for data, tags, and telemetry */
        code, pre, .mono-data, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
            font-family: {FONT_MONO} !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_PAPER_MUTED} !important;
            border-right: 1.5px solid {COLOR_RULE} !important;
        }}

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {COLOR_INK} !important;
        }}

        /* Custom Cards & Inspection Containers */
        .paper-card {{
            background-color: {COLOR_PAPER_CARD};
            border: 1.5px solid {COLOR_RULE};
            border-radius: 4px;
            padding: 18px 20px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 0px rgba(18, 24, 26, 0.08);
        }}

        .paper-card-header {{
            font-size: 13px;
            font-family: {FONT_MONO};
            font-weight: 600;
            color: {COLOR_RULE};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid {COLOR_RULE}44;
            padding-bottom: 6px;
        }}

        /* 3-Phase Flow Stage Banner */
        .process-flow-banner {{
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr;
            align-items: center;
            background-color: {COLOR_PAPER_CARD};
            border: 1.5px solid {COLOR_INK};
            border-radius: 4px;
            padding: 14px 20px;
            margin-bottom: 24px;
            box-shadow: 3px 3px 0px {COLOR_INK};
        }}

        .flow-step {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .flow-step-num {{
            font-family: {FONT_MONO};
            font-size: 11px;
            font-weight: 700;
            color: {COLOR_RULE};
            letter-spacing: 0.05em;
        }}

        .flow-step-title {{
            font-family: {FONT_INTERFACE};
            font-size: 15px;
            font-weight: 700;
            color: {COLOR_INK};
            text-transform: uppercase;
            letter-spacing: -0.01em;
        }}

        .flow-step-status {{
            font-family: {FONT_MONO};
            font-size: 12px;
            color: {COLOR_INK_MUTED};
        }}

        .flow-arrow {{
            font-family: {FONT_MONO};
            font-size: 20px;
            font-weight: 700;
            color: {COLOR_RULE};
            padding: 0 14px;
            user-select: none;
        }}

        /* Buttons & Interactive Controls */
        .stButton button {{
            font-family: {FONT_INTERFACE} !important;
            font-weight: 600 !important;
            border-radius: 4px !important;
            border: 1.5px solid {COLOR_INK} !important;
            transition: all 0.15s ease !important;
            letter-spacing: -0.01em !important;
        }}

        .stButton button[kind="primary"] {{
            background-color: {COLOR_INK} !important;
            color: {COLOR_PAPER} !important;
            box-shadow: 2px 2px 0px {COLOR_RULE} !important;
        }}

        .stButton button[kind="primary"]:hover {{
            background-color: {COLOR_INK_MUTED} !important;
            color: {COLOR_PAPER} !important;
            box-shadow: 3px 3px 0px {COLOR_INK} !important;
        }}

        .stButton button[kind="secondary"] {{
            background-color: {COLOR_PAPER_CARD} !important;
            color: {COLOR_INK} !important;
            box-shadow: 2px 2px 0px {COLOR_RULE} !important;
        }}

        .stButton button[kind="secondary"]:hover {{
            background-color: {COLOR_PAPER_MUTED} !important;
            color: {COLOR_INK} !important;
        }}

        /* Streamlit Inputs & Text Areas */
        .stTextArea textarea, .stTextInput input {{
            background-color: {COLOR_PAPER_CARD} !important;
            color: {COLOR_INK} !important;
            font-family: {FONT_MONO} !important;
            font-size: 13px !important;
            border: 1.5px solid {COLOR_RULE} !important;
            border-radius: 4px !important;
        }}

        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: {COLOR_INK} !important;
            box-shadow: 0 0 0 1px {COLOR_INK} !important;
        }}

        /* Metrics */
        [data-testid="stMetric"] {{
            background-color: {COLOR_PAPER_CARD} !important;
            border: 1.5px solid {COLOR_RULE} !important;
            border-radius: 4px !important;
            padding: 10px 14px !important;
            box-shadow: 2px 2px 0px rgba(18, 24, 26, 0.05) !important;
        }}

        [data-testid="stMetricLabel"] {{
            font-family: {FONT_INTERFACE} !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            color: {COLOR_RULE} !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLOR_INK} !important;
            font-weight: 700 !important;
            font-size: 22px !important;
        }}

        /* Status and Action Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 9px;
            border-radius: 3px;
            font-family: {FONT_MONO};
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .badge-verified, .badge-allow {{
            background-color: {COLOR_VERIFIED_GREEN}18;
            color: {COLOR_VERIFIED_GREEN};
            border: 1.5px solid {COLOR_VERIFIED_GREEN};
        }}

        .badge-redact {{
            background-color: {COLOR_REDACT_AMBER}18;
            color: {COLOR_REDACT_AMBER};
            border: 1.5px solid {COLOR_REDACT_AMBER};
        }}

        .badge-block {{
            background-color: {COLOR_BLOCK_RED}18;
            color: {COLOR_BLOCK_RED};
            border: 1.5px solid {COLOR_BLOCK_RED};
        }}

        .badge-rule, .badge-review {{
            background-color: {COLOR_RULE}18;
            color: {COLOR_RULE};
            border: 1.5px solid {COLOR_RULE};
        }}

        /* Split Screen Container Architecture */
        .split-paper-container {{
            background-color: {COLOR_PAPER_CARD};
            color: {COLOR_INK};
            border: 1.5px solid {COLOR_RULE};
            border-radius: 4px;
            padding: 16px 18px;
            min-height: 480px;
            box-shadow: 3px 3px 0px rgba(18, 24, 26, 0.08);
            position: relative;
            overflow: hidden;
        }}

        .split-dark-container {{
            background-color: {COLOR_INK};
            color: {COLOR_PAPER};
            border: 1.5px solid {COLOR_RULE};
            border-radius: 4px;
            padding: 16px 18px;
            min-height: 480px;
            box-shadow: 3px 3px 0px rgba(18, 24, 26, 0.2);
            position: relative;
        }}

        .split-gate-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 480px;
            padding: 10px 4px;
            background-color: {COLOR_PAPER_MUTED};
            border: 1.5px solid {COLOR_RULE};
            border-radius: 4px;
            box-shadow: 2px 2px 0px rgba(18, 24, 26, 0.05);
        }}

        /* Laser Scan Sweep Animation (The ONLY major animation) */
        @keyframes scan-sweep-beam {{
            0% {{
                top: 2%;
                opacity: 0.6;
            }}
            50% {{
                top: 96%;
                opacity: 1;
            }}
            100% {{
                top: 2%;
                opacity: 0.6;
            }}
        }}

        .scan-beam-active {{
            position: absolute;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent 0%, {COLOR_VERIFIED_GREEN} 20%, #ffffff 50%, {COLOR_VERIFIED_GREEN} 80%, transparent 100%);
            box-shadow: 0 0 14px {COLOR_VERIFIED_GREEN}, 0 0 4px #ffffff;
            animation: scan-sweep-beam 2.4s ease-in-out infinite;
            z-index: 25;
            pointer-events: none;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: transparent;
            border-bottom: 1.5px solid {COLOR_RULE};
        }}

        .stTabs [data-baseweb="tab"] {{
            font-family: {FONT_INTERFACE} !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            color: {COLOR_RULE} !important;
            background-color: transparent !important;
            border-radius: 4px 4px 0 0 !important;
            padding: 8px 16px !important;
            border: 1px solid transparent !important;
        }}

        .stTabs [aria-selected="true"] {{
            color: {COLOR_INK} !important;
            background-color: {COLOR_PAPER_CARD} !important;
            border: 1.5px solid {COLOR_RULE} !important;
            border-bottom: 2px solid {COLOR_PAPER_CARD} !important;
            margin-bottom: -1.5px;
        }}
    </style>
    """
