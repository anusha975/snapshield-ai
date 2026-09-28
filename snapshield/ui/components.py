"""Reusable UI Component System for SnapShield AI.

Adheres strictly to the SnapShield AI Design Language:
PRODUCT: SnapShield AI
TAGLINE: Your local privacy checkpoint for AI.

COLOR SYSTEM:
- Ink: #12181A
- Paper: #F2EEE2
- Verified Green: #2E7D5B
- Redact Amber: #C97A2B
- Block Red: #B23A3A
- Rule: #5A6B70

TYPOGRAPHY:
- Space Grotesk: Interface text, labels, titles, structural hierarchy
- IBM Plex Mono: Data, detection values, tags, performance telemetry, tokens, hashes

Architectural 3-Phase Pipeline Representation:
RAW DOCUMENT -> CHECKPOINT -> PROTECTED OUTPUT
"""

from __future__ import annotations

import textwrap
from typing import Any, List, Optional
import streamlit as st

from snapshield.core.types import Finding, PolicyAction, RiskLevel
from snapshield.npu.device import HardwareCapabilityReport
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


def st_html(markup: str) -> None:
    """Renders HTML safely into Streamlit without markdown indentation code-block artifacts."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_hardware_badge(hw: HardwareCapabilityReport) -> None:
    """Renders factual hardware execution status badge with IBM Plex Mono font."""
    if hw.is_accelerated and hw.qnn_provider_available:
        bg_col = COLOR_VERIFIED_GREEN
        text_col = COLOR_PAPER
        label = "⚡ Snapdragon NPU (QNN Active)"
    elif hw.is_accelerated and hw.directml_provider_available:
        bg_col = COLOR_RULE
        text_col = COLOR_PAPER
        label = "⚡ DirectML GPU Accelerated"
    else:
        bg_col = COLOR_PAPER_MUTED
        text_col = COLOR_INK
        label = "🛡️ Verified CPU Fallback"

    st.sidebar.markdown(
        f"""
        <div style="padding: 9px 12px; border-radius: 4px; font-family: {FONT_MONO};
                    font-size: 11px; font-weight: 600; margin-bottom: 14px;
                    background-color: {bg_col}; color: {text_col};
                    border: 1.5px solid {COLOR_INK}; text-align: center;
                    letter-spacing: 0.02em;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_scorecard(score: float, level: RiskLevel | str, findings_count: int) -> None:
    """Renders high-impact privacy risk metric cards adhering to color specification."""
    level_str = level.value if hasattr(level, "value") else str(level)
    level_upper = level_str.upper()

    color_map = {
        "LOW": COLOR_VERIFIED_GREEN,
        "MEDIUM": COLOR_REDACT_AMBER,
        "HIGH": COLOR_REDACT_AMBER,
        "CRITICAL": COLOR_BLOCK_RED,
    }
    level_color = color_map.get(level_upper, COLOR_RULE)

    cols = st.columns(3)
    with cols[0]:
        st.metric(label="Calculated Risk Score", value=f"{score:.1f} / 100")
    with cols[1]:
        st.markdown(
            f"""
            <div style="padding-top: 6px;">
                <span style="font-family: {FONT_INTERFACE}; font-size: 12px; color: {COLOR_RULE};
                             text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">
                    Risk Tier
                </span><br/>
                <span style="display: inline-block; margin-top: 5px; padding: 4px 12px; border-radius: 3px;
                             background-color: {level_color}18; color: {level_color}; font-weight: 700;
                             border: 1.5px solid {level_color}; font-family: {FONT_MONO}; font-size: 15px;">
                    {level_upper}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.metric(label="Sensitive Entities", value=str(findings_count))


def render_action_badge_html(action: PolicyAction | str) -> str:
    """Renders accessible action badge with explicit icon and label without relying on color alone.

    Icons:
    ✓ ALLOW  (Green: #2E7D5B)
    ⚠ REDACT (Amber: #C97A2B)
    ✕ BLOCK  (Red: #B23A3A)
    ◆ REVIEW (Rule: #5A6B70)
    """
    act_str = action.value if hasattr(action, "value") else str(action or "REVIEW")
    act_upper = act_str.upper()

    badge_map = {
        "ALLOW": (COLOR_VERIFIED_GREEN, "✓ ALLOW"),
        "REDACT": (COLOR_REDACT_AMBER, "⚠ REDACT"),
        "BLOCK": (COLOR_BLOCK_RED, "✕ BLOCK"),
        "REVIEW": (COLOR_RULE, "◆ REVIEW"),
    }
    color, text = badge_map.get(act_upper, (COLOR_RULE, f"◆ {act_upper}"))
    return (
        f'<span class="badge" style="background-color: {color}22; color: {color}; '
        f'border: 1.5px solid {color}; font-weight: 700; font-family: {FONT_MONO};">'
        f'{text}</span>'
    )


def get_compact_runtime_info(
    hw_report: Optional[HardwareCapabilityReport] = None,
    scan_result: Optional[Any] = None,
    benchmark_suite: Optional[Any] = None,
) -> dict:
    """Extracts truthful local AI runtime and benchmark metrics without fabrication.

    Invariants:
    - Never hardcodes benchmark values.
    - Never says 'NPU' unless the runtime confirms the relevant accelerator/provider (QNNExecutionProvider).
    - If unavailable, displays a truthful message rather than hiding the state.
    """
    from snapshield.models import DEFAULT_MODEL_FILE, is_default_model_installed
    from snapshield.npu.runtime import (
        ACCELERATED_PROVIDERS,
        NPU_PROVIDERS,
        detect_environment,
        get_active_provider,
        get_available_providers,
    )

    env = detect_environment()
    ort_version = env.get("onnxruntime_version", "unavailable")
    ort_available = ort_version != "unavailable"
    model_installed = is_default_model_installed()
    available_providers = env.get("available_providers", [])

    # 1. Model Name: Factual model on disk
    if model_installed:
        model_name = "bert-small-pii-detection-int8"
    elif DEFAULT_MODEL_FILE.exists():
        model_name = DEFAULT_MODEL_FILE.stem
    else:
        model_name = "Unavailable (Model not found)"

    # 2. Execution Provider: Factual active execution provider
    if not ort_available:
        execution_provider = "Unavailable"
    elif hw_report is not None:
        execution_provider = hw_report.active_execution_provider
    else:
        execution_provider = get_active_provider()

    # 3. Status: Verified / CPU fallback / unavailable
    if not ort_available or (not model_installed and not DEFAULT_MODEL_FILE.exists()):
        status = "unavailable"
    elif execution_provider in ACCELERATED_PROVIDERS and execution_provider != "CPUExecutionProvider":
        status = "Verified"
    else:
        status = "CPU fallback"

    # 4. Inference: Actual latency measured during scan
    inference_latency = "--"
    if scan_result is not None:
        timings = getattr(scan_result, "timings", None)
        if timings is not None:
            ai_ms = getattr(timings, "ai_inference_ms", 0.0)
            if ai_ms > 0:
                inference_latency = f"{ai_ms:.2f} ms"

    # 5. Benchmark: CPU and Accelerated (measured or unavailable)
    cpu_benchmark = "--"
    accel_benchmark = "--"

    has_accel_candidate = any(
        p in available_providers for p in ACCELERATED_PROVIDERS if p != "CPUExecutionProvider"
    )

    if benchmark_suite is not None:
        # Factual measured CPU benchmark
        if hasattr(benchmark_suite, "cpu_result") and benchmark_suite.cpu_result:
            cpu_benchmark = f"{benchmark_suite.cpu_result.mean_latency_ms:.2f} ms"

        # Factual measured Accelerated benchmark or unavailable
        if getattr(benchmark_suite, "is_accelerated_available", False) and getattr(benchmark_suite, "accelerated_result", None):
            accel_benchmark = f"{benchmark_suite.accelerated_result.mean_latency_ms:.2f} ms"
        else:
            accel_benchmark = "Accelerated execution: Not available"
    else:
        if not has_accel_candidate:
            accel_benchmark = "Accelerated execution: Not available"

    return {
        "model": model_name,
        "execution": execution_provider,
        "status": status,
        "inference": inference_latency,
        "cpu_benchmark": cpu_benchmark,
        "accelerated_benchmark": accel_benchmark,
        "is_accelerated": status == "Verified",
        "has_accel_candidate": has_accel_candidate,
        "ort_version": ort_version,
    }


def render_compact_runtime_panel(
    hw_report: Optional[HardwareCapabilityReport] = None,
    scan_result: Optional[Any] = None,
    benchmark_suite: Optional[Any] = None,
    show_benchmark_trigger: bool = True,
) -> None:
    """Renders a compact, high-density local AI runtime panel meeting the exact specification:

    LOCAL AI RUNTIME

    Model:
    [actual model]

    Execution:
    [actual provider]

    Status:
    Verified / CPU fallback / unavailable

    Inference:
    [actual latency]

    Benchmark:

    CPU:
    [actual]

    Accelerated:
    [actual or unavailable]
    """
    info = get_compact_runtime_info(
        hw_report=hw_report,
        scan_result=scan_result,
        benchmark_suite=benchmark_suite,
    )

    # Status color styling
    if info["status"] == "Verified":
        status_color = COLOR_VERIFIED_GREEN
        status_tag = "✓ Verified"
    elif info["status"] == "CPU fallback":
        status_color = COLOR_REDACT_AMBER
        status_tag = "🛡️ CPU fallback"
    else:
        status_color = COLOR_BLOCK_RED
        status_tag = "✕ unavailable"

    accel_is_unavail = "Not available" in info["accelerated_benchmark"] or info["accelerated_benchmark"] == "Unavailable"
    accel_color = COLOR_RULE if accel_is_unavail else COLOR_VERIFIED_GREEN

    st_html(
        f"""
        <div style="background-color: {COLOR_PAPER_CARD}; border: 1.5px solid {COLOR_INK}; border-radius: 4px; padding: 14px 16px; margin-bottom: 16px; box-shadow: 2px 2px 0px rgba(18, 24, 26, 0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1.5px solid {COLOR_INK}; padding-bottom: 6px; margin-bottom: 12px;">
                <span style="font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; letter-spacing: 0.05em; text-transform: uppercase;">
                    LOCAL AI RUNTIME
                </span>
                <span style="font-family: {FONT_MONO}; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 2px; background-color: {status_color}18; color: {status_color}; border: 1px solid {status_color};">
                    {status_tag}
                </span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; font-family: {FONT_MONO}; font-size: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="color: {COLOR_RULE}; font-weight: 600;">Model:</span>
                    <span style="font-weight: 600; color: {COLOR_INK}; text-align: right;">{info['model']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="color: {COLOR_RULE}; font-weight: 600;">Execution:</span>
                    <span style="font-weight: 600; color: {COLOR_INK}; text-align: right;">{info['execution']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="color: {COLOR_RULE}; font-weight: 600;">Status:</span>
                    <span style="font-weight: 700; color: {status_color}; text-align: right;">{info['status']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="color: {COLOR_RULE}; font-weight: 600;">Inference:</span>
                    <span style="font-weight: 600; color: {COLOR_INK}; text-align: right;">{info['inference']}</span>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed {COLOR_RULE}55;">
                <div style="font-family: {FONT_INTERFACE}; font-size: 12px; font-weight: 700; color: {COLOR_INK}; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.03em;">
                    Benchmark:
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px; font-family: {FONT_MONO}; font-size: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">CPU:</span>
                        <span style="font-weight: 600; color: {COLOR_INK}; text-align: right;">{info['cpu_benchmark']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Accelerated:</span>
                        <span style="font-weight: 600; color: {accel_color}; text-align: right;">{info['accelerated_benchmark']}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    )


def render_runtime_indicator(hw_report: HardwareCapabilityReport) -> None:
    """Renders prominent, truthful runtime indicator displaying active execution provider."""
    if hw_report.is_accelerated and hw_report.qnn_provider_available:
        border_col = COLOR_VERIFIED_GREEN
        text = "⚡ Snapdragon NPU (QNN Active)"
    elif hw_report.is_accelerated and hw_report.directml_provider_available:
        border_col = COLOR_VERIFIED_GREEN
        text = "⚡ DirectML GPU Accelerated"
    else:
        border_col = COLOR_RULE
        text = f"🛡️ {hw_report.active_execution_provider} (Verified CPU)"

    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
                    background-color: {COLOR_PAPER_CARD}; border: 1.5px solid {border_col};
                    border-radius: 4px; font-family: {FONT_MONO}; font-size: 12px; font-weight: 600;
                    color: {COLOR_INK}; box-shadow: 2px 2px 0px rgba(18, 24, 26, 0.06);">
            <span style="color: {border_col}; font-size: 14px;">●</span>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_split_checkpoint_gate(
    scan_result: Optional[Any],
    policy_mode: str,
    is_scanning: bool = False,
) -> None:
    """Renders the central vertical checkpoint/scan gate linking RAW DOCUMENT to PROTECTED OUTPUT."""
    action_html = ""
    gate_state_text = "GATE IDLE"
    gate_color = COLOR_RULE

    if is_scanning:
        gate_state_text = "SCANNING"
        gate_color = COLOR_VERIFIED_GREEN
    elif scan_result is not None:
        overall_action = getattr(scan_result, "overall_action", PolicyAction.REVIEW)
        action_html = render_action_badge_html(overall_action)
        gate_state_text = "EVALUATED"
        gate_color = COLOR_VERIFIED_GREEN if not scan_result.is_blocked else COLOR_BLOCK_RED

    fallback_badge = '<span class="badge badge-rule">AWAITING SCAN</span>'
    active_badge = action_html if action_html else fallback_badge

    st_html(
        f"""
        <div class="split-gate-container">
            <div style="font-family: {FONT_MONO}; font-size: 10px; font-weight: 700; color: {COLOR_RULE}; letter-spacing: 0.08em; text-align: center; margin-bottom: 8px;">
                GATE CONDUIT
            </div>
            <div style="width: 2px; height: 35px; background-color: {COLOR_RULE}; margin: 0 auto 10px auto;"></div>
            <div style="padding: 12px 14px; border: 2px solid {gate_color}; border-radius: 50%; background-color: {COLOR_PAPER_CARD}; text-align: center; box-shadow: 0 0 10px {gate_color}30;">
                <span style="font-size: 22px;">🛡️</span>
            </div>
            <div style="font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; margin-top: 10px; text-align: center; text-transform: uppercase;">
                Scan Gate
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 10px; font-weight: 700; color: {gate_color}; text-align: center; margin-top: 2px;">
                [{gate_state_text}]
            </div>
            <div style="width: 2px; height: 30px; background-color: {COLOR_RULE}; margin: 10px auto;"></div>
            <div style="text-align: center; margin: 4px 0;">
                {active_badge}
            </div>
            <div style="width: 2px; height: 30px; background-color: {COLOR_RULE}; margin: 10px auto;"></div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; text-align: center; color: {COLOR_INK_MUTED};">
                POLICY PROFILE:<br/><strong>{policy_mode.upper()}</strong>
            </div>
            <div style="width: 2px; height: 35px; background-color: {COLOR_RULE}; margin: 10px auto 0 auto;"></div>
        </div>
        """
    )


def render_policy_banner(action: PolicyAction | str, rationale: str) -> None:
    """Displays policy enforcement decision banner with explicit symbols (never color alone)."""
    act_str = action.value if hasattr(action, "value") else str(action or "REVIEW")
    act_upper = act_str.upper()

    banner_config = {
        "ALLOW": (COLOR_VERIFIED_GREEN, "✓ ALLOW: CONTENT PERMITTED", "Safe for downstream AI egress."),
        "REDACT": (COLOR_REDACT_AMBER, "⚠ REDACT: SANITIZATION MANDATORY", "Sensitive tokens scrubbed before transmission."),
        "REVIEW": (COLOR_RULE, "◆ REVIEW: MANUAL APPROVAL REQUIRED", "Explicit manual security approval required."),
        "BLOCK": (COLOR_BLOCK_RED, "✕ BLOCK: OUTRIGHT PROHIBITED", "Transmission prohibited under active policy."),
    }

    color, title, default_sub = banner_config.get(act_upper, (COLOR_RULE, "◆ POLICY ACTION", ""))
    st.markdown(
        f"""
        <div style="padding: 12px 16px; border-radius: 4px; margin: 14px 0;
                    background: {color}14; border-left: 5px solid {color};
                    border-top: 1px solid {color}40; border-right: 1px solid {color}40;
                    border-bottom: 1px solid {color}40;">
            <div style="font-family: {FONT_INTERFACE}; font-weight: 700; font-size: 14px;
                        color: {color}; letter-spacing: 0.02em;">
                {title}
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 12px; color: {COLOR_INK};
                        margin-top: 4px; line-height: 1.4;">
                {rationale or default_sub}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_findings_table(
    findings: List[Any],
    decisions: Optional[List[Any]] = None,
) -> None:
    """Renders structured Detection Panel matching specification:
    Type | Severity | Confidence | Method | Action | Masked Preview

    Example:
    AADHAAR             | HIGH | Validated | Regex + Checksum   | ⚠ REDACT | 9999 **** 0019
    MEDICAL_INFORMATION | HIGH | 94%       | Local AI           | ⚠ REDACT | [REDACTED: MEDICAL]

    Guarantees: Zero raw sensitive values displayed by default.
    """
    if not findings:
        st.info("No sensitive entities discovered in inspected content.")
        return

    # Map entity_type to policy decision action if decisions provided
    decision_map = {}
    if decisions:
        for dec in decisions:
            ent = getattr(dec, "entity_type", "")
            act = getattr(dec, "action", None)
            act_str = act.value if hasattr(act, "value") else str(act or "REDACT")
            decision_map[ent.upper()] = act_str.upper()

    table_data = []
    for f in findings:
        if hasattr(f, "entity_type"):
            ent_type = f.entity_type
            severity_str = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
            conf_val = float(getattr(f, "confidence", 1.0))
            detector_name = getattr(f, "detector", "regex").lower()
            is_valid = getattr(f, "valid", True)
            masked_preview = f.get_masked_display_value() if hasattr(f, "get_masked_display_value") else "[REDACTED]"

            # Format Confidence: 'Validated' for deterministic checksums, or percentage like '94%' for AI
            if is_valid and (detector_name in ("verhoeff", "luhn") or (detector_name == "regex" and ent_type in ("AADHAAR", "CREDIT_CARD"))):
                confidence_str = "Validated"
            elif detector_name in ("ai", "local_ai", "onnx"):
                confidence_str = f"{int(conf_val * 100)}%"
            elif is_valid and conf_val >= 0.99:
                confidence_str = "Validated"
            else:
                confidence_str = f"{int(conf_val * 100)}%"

            # Format Method
            if detector_name in ("verhoeff", "luhn") or (is_valid and ent_type in ("AADHAAR", "CREDIT_CARD")):
                method_str = "Regex + Checksum"
            elif detector_name in ("ai", "local_ai", "onnx"):
                method_str = "Local AI"
            elif detector_name in ("secret", "credentials", "entropy"):
                method_str = "High-Entropy Secrets"
            elif detector_name in ("regex", "structured"):
                method_str = "Regex Pattern"
            else:
                method_str = "Deterministic"

            # Format Action with clear icons (never color alone)
            act_val = decision_map.get(ent_type.upper(), "REDACT")
            action_icon_map = {
                "ALLOW": "✓ ALLOW",
                "REDACT": "⚠ REDACT",
                "BLOCK": "✕ BLOCK",
                "REVIEW": "◆ REVIEW",
            }
            action_display = action_icon_map.get(act_val, f"⚠ {act_val}")

            table_data.append({
                "Type": ent_type,
                "Severity": severity_str.upper(),
                "Confidence": confidence_str,
                "Method": method_str,
                "Action": action_display,
                "Masked Preview": masked_preview,
            })

        elif hasattr(f, "safe_dict"):
            d = f.safe_dict()
            table_data.append({
                "Type": d.get("entity_type"),
                "Severity": str(d.get("severity", "MEDIUM")).upper(),
                "Confidence": "Validated" if d.get("valid") else f"{int(float(d.get('confidence', 1.0)) * 100)}%",
                "Method": "Regex + Checksum" if d.get("valid") else "Pattern Match",
                "Action": "⚠ REDACT",
                "Masked Preview": d.get("masked_value", "[REDACTED]"),
            })

    st.dataframe(table_data, use_container_width=True, hide_index=True)


# =========================================================================
# 3-Phase Pipeline Architectural Components:
# RAW DOCUMENT -> CHECKPOINT -> PROTECTED OUTPUT
# =========================================================================

def render_raw_document_stage(
    doc_type: str,
    raw_content: str,
    file_name: str = "document.txt",
    image_bytes: Optional[bytes] = None,
    raw_length: int = 0,
) -> None:
    """Visual component for PHASE 01: RAW DOCUMENT."""
    st.markdown(
        f"""
        <div class="paper-card">
            <div class="paper-card-header">
                <span>01. RAW DOCUMENT</span>
                <span class="mono-data">{doc_type.upper()} • {file_name}</span>
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE}; margin-bottom: 8px;">
                INTAKE INVENTORY: {raw_length or len(raw_content)} chars • RAW UNFILTERED CONTENT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if image_bytes:
        st.image(image_bytes, caption=f"Raw Image Intake: {file_name}", use_container_width=True)
    else:
        st.text_area(
            "Raw Ingested Stream",
            value=raw_content,
            height=160,
            disabled=True,
            help="Raw source content held strictly in local volatile memory. Never sent to cloud.",
        )


def render_checkpoint_stage(
    scan_result: Any,
    policy_mode: str,
    hw_report: HardwareCapabilityReport,
) -> None:
    """Visual component for PHASE 02: CHECKPOINT."""
    st.markdown(
        f"""
        <div class="paper-card">
            <div class="paper-card-header">
                <span>02. PRIVACY CHECKPOINT</span>
                <span class="mono-data">PROFILE: {policy_mode.upper()}</span>
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE}; margin-bottom: 8px;">
                ON-DEVICE ENFORCEMENT: {hw_report.active_execution_provider}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk = getattr(scan_result, "risk_assessment", None)
    if risk:
        render_risk_scorecard(score=risk.score, level=risk.level, findings_count=risk.findings_count)

    render_policy_banner(
        action=getattr(scan_result, "overall_action", PolicyAction.REVIEW),
        rationale=f"Evaluated against '{policy_mode}' policy profile with on-device risk assessment.",
    )

    st.markdown(f"<div style='font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; margin-top: 10px;'>Discovered Entity Manifest</div>", unsafe_allow_html=True)
    detections = getattr(scan_result, "detections", [])
    decisions = getattr(scan_result, "decisions", [])
    render_findings_table(detections, decisions=decisions)


def render_protected_output_stage(
    scan_result: Any,
    image_bytes: Optional[bytes] = None,
) -> None:
    """Visual component for PHASE 03: PROTECTED OUTPUT."""
    st.markdown(
        f"""
        <div class="paper-card">
            <div class="paper-card-header">
                <span>03. PROTECTED OUTPUT</span>
                <span class="mono-data">SHA-256: {getattr(scan_result, 'integrity_hash', '')[:12]}</span>
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_VERIFIED_GREEN}; font-weight: 600; margin-bottom: 8px;">
                ● DECOUPLED SANITIZATION VERIFIED • SAFE FOR DOWNSTREAM EGRESS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_annotated_image = bool(getattr(scan_result, "annotated_image_bytes", None))

    if has_annotated_image:
        tab_visual, tab_text = st.tabs(["🖼️ Visual Bounding Overlay", "📝 Sanitized Text Stream"])
        with tab_visual:
            st.image(
                scan_result.annotated_image_bytes,
                caption=f"Visual Overlay: {scan_result.annotated_regions_count} highlighted regions.",
                use_container_width=True,
            )
        with tab_text:
            st.text_area(
                "Sanitized Payload",
                value=scan_result.sanitized_text or "[BLOCKED BY CHECKPOINT]",
                height=150,
                disabled=True,
            )
    else:
        st.text_area(
            "Sanitized Payload (Safe for AI / Egress)",
            value=scan_result.sanitized_text or "[BLOCKED BY CHECKPOINT]",
            height=160,
            disabled=True,
            help="Decoupled output guaranteed free of raw sensitive tokens.",
        )


def render_benchmark_comparison(suite: Any) -> None:
    """Renders factual hardware benchmark metrics without fabricating values."""
    st.markdown(
        f"""
        <div class="paper-card">
            <div class="paper-card-header">
                <span>HARDWARE BENCHMARK TELEMETRY</span>
                <span class="mono-data">{getattr(suite, 'iterations', 10)} ITERATIONS • BURN-IN WARMUP</span>
            </div>
            <div style="font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 600; color: {COLOR_INK};">
                Workload: {getattr(suite, 'workload_name', 'BERT-Small INT8 Token Classification')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cpu, col_accel = st.columns(2)

    with col_cpu:
        st.markdown(f"<div style='font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 700; color: {COLOR_INK}; margin-bottom: 8px;'>🖥️ CPU Execution (CPUExecutionProvider)</div>", unsafe_allow_html=True)
        cpu = suite.cpu_result
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Mean Latency", f"{cpu.mean_latency_ms:.2f} ms")
            st.metric("Median Latency", f"{cpu.median_latency_ms:.2f} ms" if cpu.median_latency_ms is not None else "N/A")
            st.metric("Cold Inference", f"{cpu.first_inference_ms:.2f} ms" if cpu.first_inference_ms is not None else "N/A")
        with m_col2:
            st.metric("Throughput", f"{cpu.throughput_items_per_sec:.1f} it/s")
            st.metric("Warm Inference", f"{cpu.warm_inference_ms:.2f} ms" if cpu.warm_inference_ms is not None else "N/A")
            st.metric("Min / Max", f"{cpu.min_latency_ms:.1f} / {cpu.max_latency_ms:.1f} ms" if cpu.min_latency_ms is not None else "N/A")

    with col_accel:
        if getattr(suite, "is_accelerated_available", False) and getattr(suite, "accelerated_result", None) is not None:
            accel = suite.accelerated_result
            st.markdown(f"<div style='font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 700; color: {COLOR_VERIFIED_GREEN}; margin-bottom: 8px;'>🚀 Verified Hardware Accelerator ({accel.execution_provider})</div>", unsafe_allow_html=True)
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                st.metric("Mean Latency", f"{accel.mean_latency_ms:.2f} ms")
                st.metric("Median Latency", f"{accel.median_latency_ms:.2f} ms" if accel.median_latency_ms is not None else "N/A")
                st.metric("Cold Inference", f"{accel.first_inference_ms:.2f} ms" if accel.first_inference_ms is not None else "N/A")
            with a_col2:
                st.metric("Throughput", f"{accel.throughput_items_per_sec:.1f} it/s")
                st.metric("Warm Inference", f"{accel.warm_inference_ms:.2f} ms" if accel.warm_inference_ms is not None else "N/A")
                st.metric("Min / Max", f"{accel.min_latency_ms:.1f} / {accel.max_latency_ms:.1f} ms" if accel.min_latency_ms is not None else "N/A")
        else:
            st.markdown(f"<div style='font-family: {FONT_INTERFACE}; font-size: 14px; font-weight: 700; color: {COLOR_RULE}; margin-bottom: 8px;'>🚀 Hardware Acceleration</div>", unsafe_allow_html=True)
            st.warning("Accelerated execution: Not available")
            st.markdown(
                f"""
                <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE}; padding: 8px;
                            background: {COLOR_PAPER_MUTED}; border-radius: 4px; line-height: 1.5;">
                    No compatible NPU/GPU provider (QNNExecutionProvider or DmlExecutionProvider)
                    confirmed in active hardware environment. Zero fabricated numbers are displayed.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_external_llm_panel(
    scan_result: Any,
    client: Any,
    policy_mode: str = "Personal",
) -> None:
    """Renders prominent external cloud AI processing panel with safety boundary verification."""
    st.markdown(
        f"""
        <div class="paper-card" style="border: 2px solid {COLOR_BLOCK_RED};">
            <div class="paper-card-header" style="color: {COLOR_BLOCK_RED}; border-bottom-color: {COLOR_BLOCK_RED}40;">
                <span>NOTICE: EXTERNAL CLOUD TRANSMISSION</span>
                <span class="mono-data">PERIMETER GATEWAY</span>
            </div>
            <div style="font-family: {FONT_INTERFACE}; font-size: 13px; color: {COLOR_INK}; line-height: 1.5;">
                Data dispatched through this gateway <strong>leaves this device</strong> to be processed
                by external provider <strong>{client.provider.upper()}</strong> (model: <code>{client.model}</code>).<br/>
                All original sensitive entities are strictly kept on-device. Only decoupled, sanitized text
                clearing the mandatory safety boundary is permitted for egress.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].metric("Target Provider", client.provider.upper())
    cols[1].metric("Target Model", client.model)
    cols[2].metric(
        "Config Status",
        "Active" if client.is_configured else "Disabled (No Key)",
        delta="Key present" if client.is_configured else "Missing LLM_API_KEY",
        delta_color="normal" if client.is_configured else "inverse",
    )

    if not client.is_configured:
        st.warning(
            "⚠️ External LLM feature is gracefully disabled: `LLM_API_KEY` is not configured in the environment."
        )
        return

    if getattr(scan_result, "is_blocked", False) or (getattr(scan_result, "overall_action", None) and scan_result.overall_action.value == "BLOCK"):
        st.error(
            "🛑 EGRESS PROHIBITED BY SAFETY BOUNDARY: Content contains blocked entity types. "
            "Zero data will be transmitted to external services."
        )
        return

    requires_review = getattr(scan_result, "requires_review", False)
    review_approved = False

    if requires_review:
        st.warning(f"⚠️ REVIEW REQUIRED: Document contains findings flagged for manual review under '{policy_mode}'.")
        review_approved = st.checkbox(
            "☑️ I have inspected the findings and explicitly approve the sanitized payload for downstream egress.",
            value=False,
        )
        if not review_approved:
            st.info("Awaiting explicit user review approval before enabling downstream forwarding.")
            return

    st.markdown("##### 🛡️ Payload Prepared for Egress (Decoupled Sanitized Text):")
    st.text_area(
        "Egress Content",
        value=scan_result.sanitized_text,
        height=110,
        disabled=True,
    )

    c_btn, _ = st.columns([2, 3])
    with c_btn:
        dispatch_btn = st.button("🚀 Transmit Sanitized Content to External LLM", type="primary", use_container_width=True)

    if dispatch_btn:
        with st.spinner(f"Contacting {client.provider.upper()} through safety boundary..."):
            try:
                response = client.forward_sanitized(
                    scan_result=scan_result,
                    review_approved=review_approved or not requires_review,
                )
                if response.success:
                    st.success(f"Response received from {client.provider.upper()} in {response.latency_ms:.2f}ms")
                    st.markdown(
                        f"""
                        <div class="paper-card">
                            <div class="paper-card-header">
                                <span>EXTERNAL AI RESPONSE ({client.provider.upper()} / {client.model})</span>
                                <span class="mono-data">STATUS: 200 OK</span>
                            </div>
                            <div style="font-family: {FONT_MONO}; font-size: 13px; color: {COLOR_INK}; white-space: pre-wrap; margin-top: 6px;">
                                {response.content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Downstream call failed: {response.error}")
            except Exception as exc:
                st.error(f"Safety boundary rejection: {exc}")


def render_privacy_report(
    scan_result: Any,
    llm_requested: bool = False,
    model_name: Optional[str] = None,
) -> None:
    """Renders the polished Privacy Audit & Compliance Report meeting the exact specification:

    Show:
    ENTITIES DETECTED
    ENTITIES REDACTED
    ENTITIES BLOCKED
    ENTITIES ALLOWED
    REVIEW REQUIRED

    RISK SCORE

    Detection breakdown:
    Regex
    Checksum
    OCR
    AI

    Performance:
    OCR
    Detection
    AI
    Total

    Runtime:
    Model
    Execution provider
    Local processing status

    External processing:
    LLM request: Yes/No

    Never include raw sensitive information.
    Allow export of a sanitized JSON report.
    """
    from snapshield.core.privacy_report import generate_sanitized_privacy_report

    report = generate_sanitized_privacy_report(
        scan_result=scan_result,
        llm_requested=llm_requested,
        model_name=model_name,
    )

    # 1. Masthead Card
    st.markdown(
        f"""
        <div class="paper-card" style="border: 2px solid {COLOR_INK}; margin-bottom: 16px;">
            <div class="paper-card-header" style="border-bottom: 1.5px solid {COLOR_INK};">
                <span style="font-family: {FONT_INTERFACE}; font-size: 15px; font-weight: 700; color: {COLOR_INK}; letter-spacing: 0.04em;">
                    🛡️ PRIVACY AUDIT & COMPLIANCE REPORT
                </span>
                <span class="mono-data">REPORT ID: {report.report_id} • SHA-256: {report.sanitized_output_hash[:10]}</span>
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_RULE}; margin-top: 4px;">
                AIR-GAPPED COMPLIANCE AUDIT • ZERO RAW SENSITIVE VALUES RETAINED
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Summary Counters Row (5 columns)
    # ENTITIES DETECTED | ENTITIES REDACTED | ENTITIES BLOCKED | ENTITIES ALLOWED | REVIEW REQUIRED
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("ENTITIES DETECTED", report.summary.entities_detected)
    with c2:
        st.metric("ENTITIES REDACTED", report.summary.entities_redacted)
    with c3:
        st.metric("ENTITIES BLOCKED", report.summary.entities_blocked)
    with c4:
        st.metric("ENTITIES ALLOWED", report.summary.entities_allowed)
    with c5:
        st.metric("REVIEW REQUIRED", report.summary.review_required)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 3. Risk Assessment Section
    # RISK SCORE
    risk_color = (
        COLOR_BLOCK_RED if report.risk.level == "CRITICAL"
        else (COLOR_REDACT_AMBER if report.risk.level in ("HIGH", "MEDIUM") else COLOR_VERIFIED_GREEN)
    )
    st.markdown(
        f"""
        <div style="background-color: {COLOR_PAPER_CARD}; border-left: 5px solid {risk_color};
                    border-top: 1.5px solid {COLOR_RULE}40; border-right: 1.5px solid {COLOR_RULE}40;
                    border-bottom: 1.5px solid {COLOR_RULE}40; border-radius: 4px;
                    padding: 12px 16px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: {FONT_INTERFACE}; font-size: 13px; font-weight: 700; color: {COLOR_INK}; text-transform: uppercase;">
                    RISK SCORE
                </span>
                <span style="font-family: {FONT_MONO}; font-size: 13px; font-weight: 700; color: {risk_color}; padding: 2px 10px; border-radius: 3px; border: 1.5px solid {risk_color}; background-color: {risk_color}15;">
                    {report.risk.level} • {report.risk.score:.1f} / 100
                </span>
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 12px; color: {COLOR_INK}; margin-top: 6px;">
                {' • '.join(report.risk.reasons) if report.risk.reasons else 'Document content evaluated clean under active policy profile.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Two Columns: Detection Breakdown & Performance
    col_breakdown, col_perf = st.columns(2, gap="medium")

    with col_breakdown:
        st.markdown(
            f"""
            <div class="paper-card" style="height: 100%;">
                <div class="paper-card-header">
                    <span>Detection breakdown:</span>
                    <span class="mono-data">EXTRACTION METHODS</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-family: {FONT_MONO}; font-size: 12px; margin-top: 6px;">
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Regex:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.detection_breakdown.regex}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Checksum:</span>
                        <span style="font-weight: 700; color: {COLOR_VERIFIED_GREEN};">{report.detection_breakdown.checksum}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">OCR:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.detection_breakdown.ocr}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">AI:</span>
                        <span style="font-weight: 700; color: {COLOR_REDACT_AMBER};">{report.detection_breakdown.ai}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_perf:
        st.markdown(
            f"""
            <div class="paper-card" style="height: 100%;">
                <div class="paper-card-header">
                    <span>Performance:</span>
                    <span class="mono-data">PIPELINE LATENCY</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-family: {FONT_MONO}; font-size: 12px; margin-top: 6px;">
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">OCR:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.performance.ocr_ms:.2f} ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Detection:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.performance.detection_ms:.2f} ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">AI:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.performance.ai_inference_ms:.2f} ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <span style="color: {COLOR_RULE}; font-weight: 700;">Total:</span>
                        <span style="font-weight: 700; color: {COLOR_VERIFIED_GREEN};">{report.performance.total_ms:.2f} ms</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 5. Two Columns: Runtime & External Processing
    col_rt, col_ext = st.columns(2, gap="medium")

    with col_rt:
        st.markdown(
            f"""
            <div class="paper-card" style="height: 100%;">
                <div class="paper-card-header">
                    <span>Runtime:</span>
                    <span class="mono-data">LOCAL HARDWARE</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-family: {FONT_MONO}; font-size: 12px; margin-top: 6px;">
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Model:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.runtime.model}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Execution provider:</span>
                        <span style="font-weight: 700; color: {COLOR_INK};">{report.runtime.execution_provider}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">Local processing status:</span>
                        <span style="font-weight: 700; color: {COLOR_VERIFIED_GREEN};">{report.runtime.local_processing_status}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ext:
        ext_color = COLOR_REDACT_AMBER if report.external_processing.llm_request == "Yes" else COLOR_VERIFIED_GREEN
        st.markdown(
            f"""
            <div class="paper-card" style="height: 100%;">
                <div class="paper-card-header">
                    <span>External processing:</span>
                    <span class="mono-data">PERIMETER AUDIT</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-family: {FONT_MONO}; font-size: 12px; margin-top: 6px;">
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed {COLOR_RULE}40;">
                        <span style="color: {COLOR_RULE}; font-weight: 600;">LLM request:</span>
                        <span style="font-weight: 700; color: {ext_color};">{report.external_processing.llm_request}</span>
                    </div>
                    <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_INK}; line-height: 1.5; padding-top: 4px;">
                        {report.external_processing.details}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 6. Export Sanitized JSON Report Button
    json_str = report.to_json(indent=2)
    st.download_button(
        label="📥 Export Sanitized JSON Report",
        data=json_str,
        file_name=f"snapshield_privacy_report_{report.report_id}.json",
        mime="application/json",
        help="Exports full audit report guaranteed free of raw sensitive information.",
        use_container_width=True,
    )


# =========================================================================
# Application-Wide User-Friendly Error Component
# =========================================================================

def render_user_friendly_error(
    exc: Exception,
    debug_mode: bool = False,
    custom_title: Optional[str] = None,
) -> None:
    """Renders a polished, actionable error panel without raw Python stack traces for normal users.

    Provides:
    - Clean SnapShield visual alert card.
    - Plain-language explanation and recovery guidance.
    - System graceful degradation status.
    - Controlled developer diagnostics when debug_mode is True.
    """
    from snapshield.core.exceptions import classify_error
    diag = classify_error(exc)
    title = custom_title or diag.title

    st.markdown(
        f"""
        <div style="background-color: {COLOR_PAPER_CARD}; border: 2px solid {COLOR_BLOCK_RED};
                    border-radius: 4px; padding: 16px 20px; margin: 16px 0 20px 0;
                    box-shadow: 2px 2px 0px rgba(178, 58, 58, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-family: {FONT_INTERFACE}; font-size: 16px; font-weight: 700; color: {COLOR_BLOCK_RED}; display: flex; align-items: center; gap: 8px;">
                    <span>⚠️</span>
                    <span>{title}</span>
                </div>
                <span style="font-family: {FONT_MONO}; font-size: 11px; font-weight: 700;
                             background-color: {COLOR_BLOCK_RED}; color: {COLOR_PAPER};
                             padding: 2px 8px; border-radius: 2px; letter-spacing: 0.04em;">
                    {diag.error_code}
                </span>
            </div>
            <div style="font-family: {FONT_INTERFACE}; font-size: 14px; color: {COLOR_INK}; line-height: 1.6; margin-bottom: 10px;">
                {diag.user_message}
            </div>
            <div style="background-color: {COLOR_PAPER_MUTED}; border-left: 3px solid {COLOR_RULE};
                        padding: 8px 12px; margin-bottom: 10px; font-family: {FONT_MONO}; font-size: 12px; color: {COLOR_INK};">
                <span style="font-weight: 700; color: {COLOR_RULE};">Action:</span> {diag.recovery_action}
            </div>
            <div style="font-family: {FONT_MONO}; font-size: 11px; color: {COLOR_VERIFIED_GREEN}; font-weight: 600;">
                {diag.degradation_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controlled Developer Debug Mode Diagnostics (Hidden from regular users)
    if debug_mode:
        with st.expander("🛠️ Developer Diagnostics (Debug Mode Active)", expanded=True):
            st.markdown(
                f"""
                <div style="font-family: {FONT_MONO}; font-size: 12px; color: {COLOR_INK};">
                    <div style="margin-bottom: 6px;"><b>Exception Type:</b> <code>{type(exc).__name__}</code></div>
                    <div style="margin-bottom: 6px;"><b>Error Code:</b> <code>{diag.error_code}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if diag.technical_details:
                st.json(diag.technical_details)
            if diag.traceback_str:
                st.code(diag.traceback_str, language="python")


