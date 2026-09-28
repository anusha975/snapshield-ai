"""Workflow view components for SnapShield AI.

Adheres strictly to the SnapShield AI Design Language:
PRODUCT: SnapShield AI
TAGLINE: Your local privacy checkpoint for AI.
COLOR SYSTEM: Ink (#12181A), Paper (#F2EEE2), Verified Green (#2E7D5B),
              Redact Amber (#C97A2B), Block Red (#B23A3A), Rule (#5A6B70).
TYPOGRAPHY: Space Grotesk (interface text) + IBM Plex Mono (data, detection values, tags, performance info).

Visual Flow Representation:
RAW DOCUMENT -> CHECKPOINT -> PROTECTED OUTPUT
"""

from __future__ import annotations

from typing import Any, Optional
import streamlit as st

from snapshield.core.types import CheckpointResult, DocumentType
from snapshield.npu.device import HardwareCapabilityReport
from snapshield.ui.components import (
    render_checkpoint_stage,
    render_findings_table,
    render_policy_banner,
    render_protected_output_stage,
    render_raw_document_stage,
    render_risk_scorecard,
    st_html,
)
from snapshield.ui.theme import (
    COLOR_BLOCK_RED,
    COLOR_INK,
    COLOR_INK_MUTED,
    COLOR_PAPER,
    COLOR_PAPER_CARD,
    COLOR_PAPER_MUTED,
    COLOR_REDACT_AMBER,
    COLOR_RULE,
    COLOR_VERIFIED_GREEN,
    FONT_INTERFACE,
    FONT_MONO,
)


def render_input_tab() -> tuple[DocumentType, Optional[str], Optional[bytes]]:
    """Renders document/text input selector adhering to Paper & Ink design."""
    input_type = st.radio(
        "Select Ingestion Source",
        options=["Plain Text / Prompt", "PDF Document", "Screenshot / Image"],
        horizontal=True,
    )

    if input_type == "Plain Text / Prompt":
        user_text = st.text_area(
            "Paste prompt or content to inspect:",
            height=160,
            placeholder="e.g. Please analyze this patient record: John Doe, SSN 123-45-6789, email john@hospital.org...",
        )
        return DocumentType.RAW_TEXT, user_text, None

    elif input_type == "PDF Document":
        uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])
        if uploaded_pdf is not None:
            return DocumentType.PDF, None, uploaded_pdf.read()
        return DocumentType.PDF, None, None

    else:
        uploaded_img = st.file_uploader("Upload Image or Screenshot", type=["png", "jpg", "jpeg"])
        if uploaded_img is not None:
            return DocumentType.IMAGE, None, uploaded_img.read()
        return DocumentType.IMAGE, None, None


def render_split_screen_interaction(
    scan_result: Optional[Any],
    raw_text: str,
    doc_type: str,
    file_name: str,
    image_bytes: Optional[bytes],
    policy_mode: str,
    hw_report: HardwareCapabilityReport,
    is_scanning: bool = False,
) -> None:
    """Renders the central split-screen layout with vertical checkpoint/scan gate.

    LEFT: Paper / raw document (with optional scan sweep animation).
    CENTER: Vertical checkpoint / scan gate conduit.
    RIGHT: Dark / protected sanitized output.
    """
    col_left, col_gate, col_right = st.columns([1.15, 0.35, 1.15], gap="small")

    # -------------------------------------------------------------
    # LEFT: Paper / Raw Document
    # -------------------------------------------------------------
    with col_left:
        scan_beam_markup = '<div class="scan-beam-active"></div>' if is_scanning else ''
        char_count = len(raw_text) if raw_text else (len(image_bytes) if image_bytes else 0)

        st_html(
            f"""
            <div class="split-paper-container">
                {scan_beam_markup}
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1.5px solid {COLOR_RULE}40; padding-bottom: 6px; margin-bottom: 12px;">
                    <div>
                        <span style="font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; text-transform: uppercase;">
                            RAW DOCUMENT
                        </span>
                    </div>
                    <div>
                        <span style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE};">
                            {doc_type.upper()} • {file_name}
                        </span>
                    </div>
                </div>
                <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_INK_MUTED}; margin-bottom: 10px;">
                    INTAKE: {char_count} bytes/chars • HELD IN VOLATILE MEMORY
                </div>
            </div>
            """
        )

        if image_bytes:
            st.image(image_bytes, caption=f"Raw Visual Document: {file_name}", use_container_width=True)
        else:
            st.text_area(
                "Raw Ingested Text",
                value=raw_text,
                height=320,
                disabled=True,
                help="Raw source content held strictly on-device in volatile memory.",
                label_visibility="collapsed",
            )

    # -------------------------------------------------------------
    # CENTER: Vertical Checkpoint / Scan Gate
    # -------------------------------------------------------------
    with col_gate:
        from snapshield.ui.components import render_split_checkpoint_gate
        render_split_checkpoint_gate(
            scan_result=scan_result,
            policy_mode=policy_mode,
            is_scanning=is_scanning,
        )

    # -------------------------------------------------------------
    # RIGHT: Dark / Protected Sanitized Output
    # -------------------------------------------------------------
    with col_right:
        integrity_hash = getattr(scan_result, "integrity_hash", "") if scan_result else ""
        hash_display = f"SHA-256: {integrity_hash[:10]}..." if integrity_hash else "INTEGRITY: UNVERIFIED"

        st_html(
            f"""
            <div class="split-dark-container">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1.5px solid {COLOR_RULE}50; padding-bottom: 6px; margin-bottom: 12px;">
                    <div>
                        <span style="font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_PAPER}; text-transform: uppercase;">
                            PROTECTED OUTPUT
                        </span>
                    </div>
                    <div>
                        <span style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE};">
                            {hash_display}
                        </span>
                    </div>
                </div>
                <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_VERIFIED_GREEN}; font-weight: 600; margin-bottom: 10px;">
                    {"● DECOUPLED SANITIZATION VERIFIED" if scan_result else "○ AWAITING CHECKPOINT SCAN"}
                </div>
            </div>
            """
        )

        if scan_result is not None:
            has_annotated_image = bool(getattr(scan_result, "annotated_image_bytes", None))

            if has_annotated_image:
                tab_visual, tab_text = st.tabs(["🖼️ Visual Bounding Map", "📝 Sanitized Text Stream"])
                with tab_visual:
                    st.image(
                        scan_result.annotated_image_bytes,
                        caption=f"Visual Overlay: {scan_result.annotated_regions_count} highlighted regions.",
                        use_container_width=True,
                    )
                with tab_text:
                    st.text_area(
                        "Sanitized Egress Content",
                        value=scan_result.sanitized_text or "[BLOCKED BY POLICY]",
                        height=280,
                        disabled=True,
                        label_visibility="collapsed",
                    )
            else:
                st.text_area(
                    "Sanitized Egress Content",
                    value=scan_result.sanitized_text or "[BLOCKED BY POLICY]",
                    height=320,
                    disabled=True,
                    label_visibility="collapsed",
                    help="Decoupled sanitized output strictly scrubbed of raw sensitive entities.",
                )
        else:
            st_html(
                f"""
                <div style="padding: 40px 16px; text-align: center; font-family: {FONT_INTERFACE}; font-size: 13px; color: {COLOR_RULE}; line-height: 1.6;">
                    🔒 <strong>Air-Gapped Output Stage</strong><br/>
                    Content will populate once passed through the central checkpoint gate.<br/>
                    Zero sensitive tokens leave this host.
                </div>
                """
            )


def render_three_phase_layout(
    scan_result: Optional[Any],
    raw_text: str,
    doc_type: str,
    file_name: str,
    image_bytes: Optional[bytes],
    policy_mode: str,
    hw_report: HardwareCapabilityReport,
) -> None:
    """Renders the main visual representation:
    RAW DOCUMENT -> CHECKPOINT -> PROTECTED OUTPUT
    """
    col_raw, col_checkpoint, col_output = st.columns([1.0, 1.3, 1.1], gap="medium")

    # ---------------------------------------------------------
    # Column 1: RAW DOCUMENT
    # ---------------------------------------------------------
    with col_raw:
        render_raw_document_stage(
            doc_type=doc_type,
            raw_content=raw_text,
            file_name=file_name,
            image_bytes=image_bytes,
            raw_length=len(raw_text) if raw_text else (len(image_bytes) if image_bytes else 0),
        )

    # ---------------------------------------------------------
    # Column 2: CHECKPOINT
    # ---------------------------------------------------------
    with col_checkpoint:
        if scan_result is not None:
            render_checkpoint_stage(
                scan_result=scan_result,
                policy_mode=policy_mode,
                hw_report=hw_report,
            )
        else:
            st_html(
                f"""
                <div class="paper-card" style="opacity: 0.7;">
                    <div class="paper-card-header">
                        <span>02. PRIVACY CHECKPOINT</span>
                        <span class="mono-data">STATUS: IDLE</span>
                    </div>
                    <div style="font-family: {FONT_INTERFACE}; font-size: 13px; color: {COLOR_RULE}; line-height: 1.6;">
                        Awaiting ingestion trigger.<br/>
                        When executed, content undergoes deterministic detection, credential scanning,
                        on-device local AI inference, risk scoring, and policy enforcement.
                    </div>
                </div>
                """
            )

    # ---------------------------------------------------------
    # Column 3: PROTECTED OUTPUT
    # ---------------------------------------------------------
    with col_output:
        if scan_result is not None:
            render_protected_output_stage(
                scan_result=scan_result,
                image_bytes=image_bytes,
            )
        else:
            st_html(
                f"""
                <div class="paper-card" style="opacity: 0.7;">
                    <div class="paper-card-header">
                        <span>03. PROTECTED OUTPUT</span>
                        <span class="mono-data">EGRESS: AIR-GAPPED</span>
                    </div>
                    <div style="font-family: {FONT_INTERFACE}; font-size: 13px; color: {COLOR_RULE}; line-height: 1.6;">
                        Protected sanitized payload will be generated here.<br/>
                        Original raw sensitive tokens are decoupled in volatile memory
                        and replaced with policy-cleared tokens.
                    </div>
                </div>
                """
            )


def render_checkpoint_results(result: CheckpointResult) -> None:
    """Legacy helper maintained for test backwards compatibility."""
    tab_assessment, tab_redacted = st.tabs(["📊 Risk & Policy Assessment", "🛡️ Sanitized Output"])
    with tab_assessment:
        render_risk_scorecard(
            score=result.risk_assessment.total_score,
            level=result.risk_assessment.level,
            findings_count=result.risk_assessment.findings_count,
        )
        render_policy_banner(
            action=result.policy_decision.action,
            rationale=result.policy_decision.rationale,
        )
        render_findings_table(result.detected_findings)
    with tab_redacted:
        if result.redaction_result:
            st.text_area("Sanitized Content", value=result.redaction_result.sanitized_text, height=140, disabled=True)


def render_scan_result_view(result: Any, llm_requested: bool = False) -> None:
    """Detailed tabbed inspection view for in-depth audit, telemetry, and overlay analysis."""
    from snapshield.ui.components import render_privacy_report

    tabs = ["📄 Privacy Audit Report", "📊 Findings Manifest & Decisions", "🛡️ Sanitized Output", "⏱️ Telemetry & Timings"]
    has_image = bool(getattr(result, "annotated_image_bytes", None))
    if has_image:
        tabs.insert(2, "🖼️ Visual Overlay")

    tab_handles = st.tabs(tabs)

    # Tab 0: Polished Privacy Audit Report
    with tab_handles[0]:
        render_privacy_report(result, llm_requested=llm_requested)

    # Tab 1: Findings Manifest & Decisions
    with tab_handles[1]:
        risk = getattr(result, "risk_assessment", None)
        if risk is not None:
            render_risk_scorecard(score=risk.score, level=risk.level, findings_count=risk.findings_count)

        overall_action = getattr(result, "overall_action", None)
        reasons = []
        decisions = getattr(result, "decisions", [])
        for d in decisions:
            if hasattr(d, "reason") and d.reason:
                reasons.append(f"• **{getattr(d, 'entity_type', 'UNKNOWN')}**: {d.reason}")
        rationale_text = "\n".join(reasons) if reasons else "Content evaluated under active policy profile."
        render_policy_banner(action=overall_action, rationale=rationale_text)

        st_html(f"<div style='font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; margin-top: 10px;'>Discovered Entity Manifest</div>")
        detections = getattr(result, "detections", [])
        decisions = getattr(result, "decisions", [])
        render_findings_table(detections, decisions=decisions)

        if getattr(result, "errors", None):
            with st.expander("⚠️ Pipeline Notices & Diagnostics", expanded=False):
                for err in result.errors:
                    st.warning(err)

    # Tab 2 (optional): Visual Overlay (if present)
    if has_image:
        with tab_handles[2]:
            st_html(
                f"""
                <div style="font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 700; color: {COLOR_INK}; margin-bottom: 4px;">
                    Screenshot / Image Bounding Box Overlay
                </div>
                <div style="font-family: {FONT_MONO}; font-size: 12px; color: {COLOR_RULE}; margin-bottom: 10px;">
                    Identified {result.annotated_regions_count} sensitive regions. Color-coded: Red (Block), Amber (Redact), Rule (Review).
                </div>
                """
            )
            st.image(result.annotated_image_bytes, use_container_width=True)

    # Tab 3: Sanitized Output
    sanitized_idx = 3 if has_image else 2
    with tab_handles[sanitized_idx]:
        st_html(
            f"""
            <div style="font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 700; color: {COLOR_INK}; margin-bottom: 4px;">
                Safe Sanitized Output (Ready for Egress)
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_VERIFIED_GREEN}; font-weight: 600; margin-bottom: 8px;">
                ● INTEGRITY HASH (SHA-256): {getattr(result, 'integrity_hash', '')}
            </div>
            """
        )
        st.text_area("Sanitized Stream", value=result.sanitized_text or "[BLOCKED]", height=150, disabled=True)

    # Tab 4: Telemetry
    timing_idx = 4 if has_image else 3
    with tab_handles[timing_idx]:
        st_html(
            f"""
            <div style="font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 700; color: {COLOR_INK}; margin-bottom: 8px;">
                Pipeline Telemetry Timings (Local perf_counter)
            </div>
            """
        )
        if hasattr(result, "timings") and result.timings:
            t = result.timings
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Extraction", f"{t.extraction_ms:.2f}ms")
            c2.metric("Local OCR", f"{t.ocr_ms:.2f}ms")
            c3.metric("Structured Det.", f"{t.structured_detection_ms:.2f}ms")
            c4.metric("Contextual AI", f"{t.ai_inference_ms:.2f}ms")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Risk Engine", f"{t.risk_ms:.2f}ms")
            c6.metric("Policy Engine", f"{t.policy_ms:.2f}ms")
            c7.metric("Redactor", f"{t.redaction_ms:.2f}ms")
            c8.metric("Total Latency", f"{t.total_ms:.2f}ms")
