"""SnapShield AI - Primary Interaction & Split-Screen Architecture.

Central Visual Concept:
RAW DOCUMENT
     ↓
SCAN CHECKPOINT
     ↓
PROTECTED OUTPUT

Split-Screen Layout:
- LEFT: Paper / raw document
- CENTER: Vertical checkpoint / scan gate
- RIGHT: Dark / protected sanitized output

Action Classification (Never relying on color alone):
- Green: ✓ ALLOW
- Amber: ⚠ REDACT
- Red:   ✕ BLOCK
- Rule:  ◆ REVIEW

Major Animation:
- Scan sweep beam across the raw document during scanning.
- All other components remain visually stable.

Primary Controls:
- Scan button
- Policy selector
- AI detection toggle
- Runtime indicator
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Ensure snapshield package root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

from PIL import Image, ImageDraw
import streamlit as st

from snapshield.config.policies import get_policy_manager
from snapshield.config.settings import get_settings
from snapshield.core.llm_client import get_llm_client
from snapshield.core.models import ScanResult
from snapshield.core.types import PolicyAction
from snapshield.data import load_synthetic_samples
from snapshield.npu.benchmark import run_benchmark
from snapshield.npu.device import probe_hardware_capabilities
from snapshield.services.scan_service import scan_document, scan_image, scan_pdf

from snapshield.ui.components import (
    render_action_badge_html,
    render_benchmark_comparison,
    render_external_llm_panel,
    render_findings_table,
    render_runtime_indicator,
    st_html,
)
from snapshield.ui.layout import (
    render_header,
    render_process_flow_banner,
    render_sidebar,
    setup_page_config,
)
from snapshield.ui.theme import (
    COLOR_BLOCK_RED,
    COLOR_INK,
    COLOR_INK_MUTED,
    COLOR_PAPER,
    COLOR_PAPER_CARD,
    COLOR_REDACT_AMBER,
    COLOR_RULE,
    COLOR_VERIFIED_GREEN,
    FONT_INTERFACE,
    FONT_MONO,
)
from snapshield.ui.views import (
    render_scan_result_view,
    render_split_screen_interaction,
)


def _generate_synthetic_screenshot() -> bytes:
    """Generates an in-memory sample screenshot with synthetic test entities for 1-click evaluation."""
    img = Image.new("RGB", (680, 320), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (680, 44)], fill=(18, 24, 26))
    draw.text((20, 14), "Confidential Patient Portal - Internal Records", fill=(242, 238, 226))

    # Record content with sample test entities
    draw.text((24, 65), "Attending Physician: Dr. Robert Smith", fill=(18, 24, 26))
    draw.text((24, 105), "National Aadhaar ID: 9999 9999 0019", fill=(18, 24, 26))
    draw.text((24, 145), "Production DB: postgres://admin:secretPass123@db.internal:5432/patients", fill=(18, 24, 26))
    draw.text((24, 185), "Clinical Contact: robert.smith@apollo-hospital.org", fill=(18, 24, 26))
    draw.text((24, 225), "OpenAI Deployment Key: sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678", fill=(18, 24, 26))
    draw.text((24, 265), "Emergency Contact: +91 98765 43210", fill=(18, 24, 26))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    """Initializes and executes the primary SnapShield interaction."""
    setup_page_config()
    settings = get_settings()

    # Hardware capabilities (verified without fabrication)
    hw_report = probe_hardware_capabilities()

    # Session State management
    if "current_scan_result" not in st.session_state:
        st.session_state["current_scan_result"] = None
    if "raw_input_text" not in st.session_state:
        st.session_state["raw_input_text"] = ""
    if "raw_image_bytes" not in st.session_state:
        st.session_state["raw_image_bytes"] = None
    if "raw_pdf_bytes" not in st.session_state:
        st.session_state["raw_pdf_bytes"] = None
    if "active_doc_type" not in st.session_state:
        st.session_state["active_doc_type"] = "TEXT"
    if "active_filename" not in st.session_state:
        st.session_state["active_filename"] = "prompt.txt"
    if "is_scanning" not in st.session_state:
        st.session_state["is_scanning"] = False
    if "cached_benchmark_suite" not in st.session_state:
        st.session_state["cached_benchmark_suite"] = None


    # Render Sidebar with Compact Local AI Runtime Panel
    sidebar_config = render_sidebar(
        hw_report=hw_report,
        scan_result=st.session_state["current_scan_result"],
        benchmark_suite=st.session_state["cached_benchmark_suite"],
    )

    # Editorial Header
    render_header()

    # Available policy profiles
    policy_manager = get_policy_manager()
    available_profiles = policy_manager.list_profiles()

    # =========================================================================
    # PRIMARY CONTROL BAR:
    # Policy selector • AI detection toggle • Runtime indicator • Scan button
    # =========================================================================
    st_html(
        f"""
        <div style="background-color: {COLOR_PAPER_CARD}; border: 1.5px solid {COLOR_RULE}; border-radius: 4px; padding: 12px 18px; margin-bottom: 18px; box-shadow: 2px 2px 0px rgba(18, 24, 26, 0.06);">
            <div style="font-family: {FONT_MONO}; font-size: 11px; font-weight: 700; color: {COLOR_RULE}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                SECURITY CONTROLS & ON-DEVICE RUNTIME
            </div>
        </div>
        """
    )

    c_pol, c_ai, c_rt, c_btn = st.columns([1.3, 1.3, 1.8, 1.3], gap="medium")

    with c_pol:
        selected_policy = st.selectbox(
            "Policy Profile:",
            options=available_profiles,
            index=0,
            help="Personal: Protects PII, financial, and healthcare data. Developer: Blocks credentials, secrets, and API keys.",
        )

    with c_ai:
        ai_enabled = st.checkbox(
            "Contextual AI Detection",
            value=True,
            help="Enables on-device lightweight BERT token classification model for contextual entity detection.",
        )
        llm_forwarding = st.checkbox(
            "Allow Downstream AI Egress",
            value=False,
            help="Enables perimeter forwarding to external cloud LLM for sanitized payloads.",
        )

    with c_rt:
        st.markdown(
            f"""
            <div style="font-family: {FONT_INTERFACE}; font-size: 12px; font-weight: 600;
                        color: {COLOR_RULE}; margin-bottom: 4px; text-transform: uppercase;">
                Execution Provider:
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_runtime_indicator(hw_report)

    with c_btn:
        st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
        execute_scan = st.button(
            "🛡️ Run Privacy Checkpoint",
            type="primary",
            use_container_width=True,
            help="Executes local OCR, detection pipeline, risk scoring, and redaction.",
        )

    # Ingestion Source Intake Accordion
    has_scan = st.session_state["current_scan_result"] is not None
    with st.expander("📥 Ingestion Intake Source (Text, PDF Document, or Screenshot/Image)", expanded=not has_scan):
        c_mode, c_sample = st.columns([1.5, 2.5])
        with c_mode:
            ingestion_mode = st.radio(
                "Content Modality:",
                options=["📝 Plain Text / Prompt", "📄 PDF Document", "🖼️ Screenshot / Image"],
                horizontal=True,
            )

        with c_sample:
            if ingestion_mode == "📝 Plain Text / Prompt":
                from snapshield.data import list_sample_documents
                sample_docs = list_sample_documents()
                doc_titles = list(sample_docs.keys())

                c_doc_select, c_doc_btn = st.columns([2.2, 1.2])
                with c_doc_select:
                    chosen_doc_title = st.selectbox(
                        "Synthetic Demo Documents (Zero Real PII):",
                        options=doc_titles,
                        index=0,
                        help="Select from verified synthetic documents featuring Indian PII, medical records, and developer secrets.",
                    )
                with c_doc_btn:
                    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                    if st.button("📄 Load Demo Document", use_container_width=True):
                        doc_info = sample_docs[chosen_doc_title]
                        st.session_state["raw_input_text"] = doc_info["content"]
                        st.session_state["active_doc_type"] = "TEXT"
                        st.session_state["active_filename"] = doc_info["filename"]
                        st.session_state["raw_image_bytes"] = None
                        st.session_state["raw_pdf_bytes"] = None
                        st.session_state["current_scan_result"] = None
                        st.success(f"Loaded demo: {doc_info['filename']}")
                        st.rerun()

            elif ingestion_mode == "📄 PDF Document":
                uploaded_pdf = st.file_uploader(
                    "Upload PDF Document (.pdf):",
                    type=["pdf"],
                    help="PDF documents are parsed 100% locally on-device. Zero cloud transmission.",
                )
                if uploaded_pdf:
                    st.session_state["raw_pdf_bytes"] = uploaded_pdf.read()
                    st.session_state["raw_image_bytes"] = None
                    st.session_state["raw_input_text"] = f"[PDF Document Loaded: {uploaded_pdf.name}]"
                    st.session_state["active_doc_type"] = "PDF"
                    st.session_state["active_filename"] = uploaded_pdf.name
                    st.session_state["current_scan_result"] = None

            else:
                c_demo_btn, c_up = st.columns([1.5, 2.5])
                with c_demo_btn:
                    if st.button("📸 Load Sample Synthetic Screenshot"):
                        st.session_state["raw_image_bytes"] = _generate_synthetic_screenshot()
                        st.session_state["raw_pdf_bytes"] = None
                        st.session_state["raw_input_text"] = ""
                        st.session_state["active_doc_type"] = "IMAGE"
                        st.session_state["active_filename"] = "synthetic_medical_portal.png"
                        st.session_state["current_scan_result"] = None
                        st.rerun()
                with c_up:
                    uploaded = st.file_uploader(
                        "Upload PNG, JPG, or JPEG:",
                        type=["png", "jpg", "jpeg"],
                        help="Images are processed 100% locally with local OCR. Zero cloud egress.",
                    )
                    if uploaded:
                        st.session_state["raw_image_bytes"] = uploaded.read()
                        st.session_state["raw_pdf_bytes"] = None
                        st.session_state["raw_input_text"] = ""
                        st.session_state["active_doc_type"] = "IMAGE"
                        st.session_state["active_filename"] = uploaded.name
                        st.session_state["current_scan_result"] = None

        if ingestion_mode == "📝 Plain Text / Prompt":
            input_text = st.text_area(
                f"Document Payload [{st.session_state.get('active_filename', 'pasted_text.txt')}]:",
                value=st.session_state["raw_input_text"],
                height=140,
                placeholder="Type, paste, or click 'Load Demo Document' to inspect...",
            )
            st.session_state["raw_input_text"] = input_text
            st.session_state["active_doc_type"] = "TEXT"

    # Execution Trigger Handling
    if execute_scan:
        st.session_state["is_scanning"] = True
        st.session_state["active_error"] = None
        try:
            with st.spinner("Scanning content through local privacy checkpoint..."):
                start_time = time.perf_counter()
                if st.session_state["active_doc_type"] == "IMAGE" and st.session_state.get("raw_image_bytes"):
                    scan_res = scan_image(
                        image_input=st.session_state["raw_image_bytes"],
                        policy_mode=selected_policy,
                        ai_enabled=ai_enabled,
                        file_name=st.session_state["active_filename"],
                    )
                    st.session_state["raw_input_text"] = scan_res.original_text or "[Local OCR Extracted Text]"

                elif st.session_state["active_doc_type"] == "PDF" and st.session_state.get("raw_pdf_bytes"):
                    scan_res = scan_pdf(
                        pdf_input=st.session_state["raw_pdf_bytes"],
                        policy_mode=selected_policy,
                        ai_enabled=ai_enabled,
                        file_name=st.session_state["active_filename"],
                    )
                    st.session_state["raw_input_text"] = scan_res.original_text or "[Extracted PDF Content]"

                else:
                    text_to_scan = st.session_state["raw_input_text"]
                    if not text_to_scan.strip():
                        from snapshield.core.exceptions import EmptyFileError
                        raise EmptyFileError("Document content is empty. Please enter text or click 'Load Demo Document'.")
                    scan_res = scan_document(
                        document=text_to_scan,
                        policy_mode=selected_policy,
                        ai_enabled=ai_enabled,
                    )

                st.session_state["current_scan_result"] = scan_res
                st.session_state["active_error"] = None
        except Exception as exc:
            st.session_state["active_error"] = exc
        finally:
            st.session_state["is_scanning"] = False


    # Render User-Friendly Error Alert if an exception occurred
    if st.session_state.get("active_error") is not None:
        from snapshield.ui.components import render_user_friendly_error
        render_user_friendly_error(
            exc=st.session_state["active_error"],
            debug_mode=st.session_state.get("debug_mode", False),
        )

    # =========================================================================
    # THE CENTRAL VISUAL CONCEPT:
    # RAW DOCUMENT (Left)  ↓  SCAN CHECKPOINT (Center)  ↓  PROTECTED OUTPUT (Right)
    # =========================================================================
    render_split_screen_interaction(
        scan_result=st.session_state["current_scan_result"],
        raw_text=st.session_state["raw_input_text"],
        doc_type=st.session_state["active_doc_type"],
        file_name=st.session_state["active_filename"],
        image_bytes=st.session_state["raw_image_bytes"],
        policy_mode=selected_policy,
        hw_report=hw_report,
        is_scanning=st.session_state["is_scanning"],
    )

    # Detailed Audit & Entity Manifest (Rendered below the split screen when results exist)
    if st.session_state["current_scan_result"] is not None:
        scan_res = st.session_state["current_scan_result"]

        st.markdown(f"<hr style='border: none; border-top: 1.5px solid {COLOR_RULE}; margin: 24px 0 16px 0;'/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-family: {FONT_INTERFACE}; font-size: 15px; font-weight: 700; color: {COLOR_INK}; margin-bottom: 12px;">
                Checkpoint Manifest & Diagnostic Telemetry
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_scan_result_view(scan_res, llm_requested=llm_forwarding)

        # External LLM perimeter forwarding (if enabled)
        if llm_forwarding:
            llm_client = get_llm_client()
            render_external_llm_panel(
                scan_result=scan_res,
                client=llm_client,
                policy_mode=selected_policy,
            )

    # Hardware Benchmark Utility
    st.markdown(f"<hr style='border: none; border-top: 1px solid {COLOR_RULE}30; margin: 28px 0 14px 0;'/>", unsafe_allow_html=True)
    with st.expander("⚡ On-Device Hardware Inference Benchmarking (CPU vs NPU/EP)", expanded=False):
        st.markdown(
            f"""
            <div style="font-family: {FONT_INTERFACE}; font-size: 13px; color: {COLOR_INK}; margin-bottom: 10px;">
                Benchmark identical token classification workloads using CPU vs verified accelerated hardware providers.
                Measures cold start, steady-state latency, throughput, and standard deviation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run Benchmark Suite (10 Iterations)", type="secondary"):
            with st.spinner("Executing benchmark across available execution providers..."):
                bench_suite = run_benchmark(iterations=10, warmup_iterations=2)
                st.session_state["cached_benchmark_suite"] = bench_suite
                render_benchmark_comparison(bench_suite)
        elif st.session_state["cached_benchmark_suite"] is not None:
            render_benchmark_comparison(st.session_state["cached_benchmark_suite"])


if __name__ == "__main__":
    main()
