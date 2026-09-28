"""Layout and theme setup for SnapShield AI Streamlit interface.

Design Language Specification:
PRODUCT: SnapShield AI
TAGLINE: Your local privacy checkpoint for AI.
COLOR SYSTEM: Ink (#12181A), Paper (#F2EEE2), Verified Green (#2E7D5B),
              Redact Amber (#C97A2B), Block Red (#B23A3A), Rule (#5A6B70).
TYPOGRAPHY: Space Grotesk (interface text) + IBM Plex Mono (data, detection values, tags, performance info).
"""

from __future__ import annotations

import streamlit as st

from snapshield.npu.device import HardwareCapabilityReport
from snapshield.ui.components import render_hardware_badge, st_html
from snapshield.ui.theme import (
    COLOR_BLOCK_RED,
    COLOR_INK,
    COLOR_PAPER,
    COLOR_PAPER_CARD,
    COLOR_REDACT_AMBER,
    COLOR_RULE,
    COLOR_VERIFIED_GREEN,
    FONT_INTERFACE,
    FONT_MONO,
    get_theme_css,
)


def setup_page_config() -> None:
    """Configures high-level Streamlit page metadata and injects editorial Paper & Ink styling."""
    st.set_page_config(
        page_title="SnapShield AI - Privacy Checkpoint",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject design system stylesheet
    st.markdown(get_theme_css(), unsafe_allow_html=True)


def render_header() -> None:
    """Renders the authoritative SnapShield AI editorial masthead."""
    st_html(
        f"""
        <div style="padding: 16px 0 12px 0; border-bottom: 2px solid {COLOR_INK}; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🛡️</span>
                        <span style="font-family: {FONT_INTERFACE}; font-size: 28px; font-weight: 700; color: {COLOR_INK}; letter-spacing: -0.03em;">
                            SnapShield AI
                        </span>
                        <span style="background-color: {COLOR_INK}; color: {COLOR_PAPER}; font-family: {FONT_MONO}; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 2px; text-transform: uppercase; letter-spacing: 0.05em; margin-left: 6px;">
                            On-Device
                        </span>
                    </div>
                    <div style="font-family: {FONT_INTERFACE}; font-size: 15px; font-weight: 500; color: {COLOR_RULE}; margin-top: 4px;">
                        Your local privacy checkpoint for AI.
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE}; letter-spacing: 0.04em;">
                        INVARIANT: ZERO RAW EGRESS
                    </div>
                    <div style="font-family: {FONT_MONO}; font-size: 12px; font-weight: 600; color: {COLOR_VERIFIED_GREEN};">
                        ● AIR-GAPPED VERIFICATION ACTIVE
                    </div>
                </div>
            </div>
        </div>
        """
    )


def render_process_flow_banner(active_stage: int = 1) -> None:
    """Visually represents the core architectural pipeline:
    RAW DOCUMENT -> CHECKPOINT -> PROTECTED OUTPUT
    """
    st_html(
        f"""
        <div class="process-flow-banner">
            <div class="flow-step" style="opacity: {1.0 if active_stage >= 1 else 0.6};">
                <span class="flow-step-num">PHASE 01</span>
                <span class="flow-step-title">RAW DOCUMENT</span>
                <span class="flow-step-status">Intake & Local OCR</span>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step" style="opacity: {1.0 if active_stage >= 2 else 0.6};">
                <span class="flow-step-num">PHASE 02</span>
                <span class="flow-step-title">CHECKPOINT</span>
                <span class="flow-step-status">Detection • Risk • Policy</span>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step" style="opacity: {1.0 if active_stage >= 3 else 0.6};">
                <span class="flow-step-num">PHASE 03</span>
                <span class="flow-step-title">PROTECTED OUTPUT</span>
                <span class="flow-step-status">Sanitized Payload & Visual Map</span>
            </div>
        </div>
        """
    )


def render_sidebar(
    hw_report: HardwareCapabilityReport,
    scan_result: Optional[Any] = None,
    benchmark_suite: Optional[Any] = None,
) -> dict:
    """Renders the configuration sidebar adhering strictly to Paper & Ink styling."""
    from snapshield.ui.components import render_compact_runtime_panel

    st.sidebar.markdown(
        "\n".join(
            line.strip()
            for line in f"""
        <div style="padding-bottom: 12px; border-bottom: 1px solid {COLOR_RULE}44; margin-bottom: 14px;">
            <div style="font-family: {FONT_INTERFACE}; font-size: 18px; font-weight: 700; color: {COLOR_INK};">
                SnapShield AI
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE}; margin-top: 2px;">
                v1.0 • Privacy Checkpoint
            </div>
        </div>
        """.splitlines()
            if line.strip()
        ),
        unsafe_allow_html=True,
    )

    # Render the Compact Local AI Runtime Panel
    render_compact_runtime_panel(
        hw_report=hw_report,
        scan_result=scan_result,
        benchmark_suite=benchmark_suite,
    )

    st.sidebar.markdown(f"<hr style='border: none; border-top: 1px solid {COLOR_RULE}44; margin: 16px 0;'/>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; text-transform: uppercase; margin-bottom: 8px;'>Security Policies</div>", unsafe_allow_html=True)

    from snapshield.config.policies import get_policy_manager
    policy_manager = get_policy_manager()
    available_profiles = policy_manager.list_profiles()

    policy_profile = st.sidebar.selectbox(
        "Active Privacy Profile",
        options=available_profiles,
        index=0,
        help="Select policy ruleset: Personal (PII, Financial, Medical) or Developer (API keys, Passwords, Code).",
    )

    enable_local_ai = st.sidebar.checkbox(
        "Enable Contextual AI Detection",
        value=True,
        help="Leverage on-device lightweight ONNX model for contextual entity detection.",
    )

    enable_llm_forwarding = st.sidebar.checkbox(
        "Allow Downstream AI Forwarding",
        value=False,
        help="Forward strictly policy-cleared, redacted output to external LLM.",
    )

    st.sidebar.markdown(f"<hr style='border: none; border-top: 1px solid {COLOR_RULE}44; margin: 16px 0;'/>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; text-transform: uppercase; margin-bottom: 8px;'>Developer Tools</div>", unsafe_allow_html=True)

    debug_mode = st.sidebar.checkbox(
        "Developer Debug Diagnostics",
        value=st.session_state.get("debug_mode", False),
        help="Enables structured technical diagnostics, error codes, and exception details.",
    )
    st.session_state["debug_mode"] = debug_mode

    return {
        "policy_profile": policy_profile,
        "enable_llm_forwarding": enable_llm_forwarding,
        "enable_local_ai": enable_local_ai,
        "debug_mode": debug_mode,
    }
