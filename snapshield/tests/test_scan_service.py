"""Comprehensive tests for the unified scan service (services/scan_service.py).

Validates:
1. Complete pipeline workflow:
   INPUT -> extraction -> OCR -> structured detection -> secret detection ->
   AI detection -> result merger -> risk engine -> policy -> redaction -> ScanResult
2. Accurate telemetry timings using time.perf_counter():
   - extraction_ms
   - ocr_ms
   - structured_detection_ms
   - ai_inference_ms
   - risk_ms
   - policy_ms
   - redaction_ms
   - total_ms
3. Clean architecture: No UI imports or dependencies.
4. Input versatility: Works with raw string or InputDocument.
5. Graceful degradation: Continues with deterministic detection if AI is unavailable,
   and exposes the unavailable state in errors and telemetry.
6. Decoupled sanitized text from original text.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.models import (
    InputDocument,
    PolicyAction,
    RiskLevel,
    ScanResult,
    ScanTiming,
)
from snapshield.services.scan_service import (
    ScanService,
    scan_document,
)


# =====================================================================
# 1. Pipeline Execution with Raw Text Input
# =====================================================================

def test_scan_document_raw_text_pipeline_personal_mode():
    """Verify full pipeline with raw text input under Personal policy mode."""
    text = (
        "CONFIDENTIAL CASE NOTES\n"
        "Patient Aadhaar: 999912345678\n"
        "Phone number: +91 9876543210\n"
        "Email address: developer@example.com\n"
        "Clinical diagnosis: Diagnosed with acute hypertension.\n"
    )

    result = scan_document(text, policy_mode="Personal", ai_enabled=True)

    assert isinstance(result, ScanResult)
    assert result.original_text == text
    assert len(result.sanitized_text) > 0
    assert result.sanitized_text != text

    # Verify PII redacted in Personal mode
    assert "999912345678" not in result.sanitized_text
    assert "[REDACTED: AADHAAR]" in result.sanitized_text
    assert "9876543210" not in result.sanitized_text
    assert "[REDACTED: PHONE]" in result.sanitized_text

    # Verify Email allowed in Personal mode
    assert "developer@example.com" in result.sanitized_text

    # Detections and decisions populated
    assert len(result.detections) >= 3
    assert len(result.decisions) == len(result.detections)

    # Risk Assessment populated
    assert result.risk_assessment.score > 0.0

    # Integrity hash calculated
    assert len(result.integrity_hash) == 64


def test_scan_document_developer_mode_blocks_credentials():
    """Verify Developer mode blocks credentials and marks is_blocked."""
    code_text = (
        "// Production environment configuration\n"
        "export OPENAI_API_KEY='sk-proj-syntheticKeyAlphaBeta1234567890abcdef'\n"
        "export DB_URL='postgres://user:SuperSecretDevPassword123@10.0.1.5:5432/core'\n"
        "export USER_EMAIL='admin@network.internal'\n"
    )

    result = scan_document(code_text, policy_mode="Developer", ai_enabled=True)

    assert isinstance(result, ScanResult)
    # Developer mode blocks API_KEY
    assert result.is_blocked is True
    assert result.overall_action == PolicyAction.BLOCK
    assert "SuperSecretDevPassword123" not in result.sanitized_text
    assert "syntheticKeyAlphaBeta" not in result.sanitized_text


# =====================================================================
# 2. Timing Telemetry Verification (time.perf_counter)
# =====================================================================

def test_timing_telemetry_metrics_measured():
    """Verify all required pipeline phase timings are accurately measured and recorded:

    extraction_ms
    ocr_ms
    structured_detection_ms
    ai_inference_ms
    risk_ms
    policy_ms
    redaction_ms
    total_ms
    """
    text = "Contact: +91 9876543210, Aadhaar: 999912345678."
    result = scan_document(text, policy_mode="Personal", ai_enabled=True)

    timings = result.timings
    assert isinstance(timings, ScanTiming)

    # All metrics must be non-negative floats
    assert timings.extraction_ms >= 0.0
    assert timings.ocr_ms >= 0.0
    assert timings.structured_detection_ms >= 0.0
    assert timings.ai_inference_ms >= 0.0
    assert timings.risk_ms >= 0.0
    assert timings.policy_ms >= 0.0
    assert timings.redaction_ms >= 0.0
    assert timings.total_ms > 0.0

    # Total must reflect the sum of stages
    assert timings.total_ms >= (
        timings.structured_detection_ms + timings.risk_ms + timings.policy_ms + timings.redaction_ms
    )


# =====================================================================
# 3. InputDocument Versatility
# =====================================================================

def test_scan_document_with_input_document_model():
    """Verify scan_document seamlessly ingests an InputDocument instance."""
    doc = InputDocument(
        document_type="RAW_TEXT",
        raw_content="Doctor notes: Dr. Robert Adams diagnosed hypertension.",
        file_name="clinical_notes.txt",
        file_size_bytes=52,
    )

    result = scan_document(doc, policy_mode="Personal", ai_enabled=True)

    assert result.original_text == doc.raw_content
    assert len(result.detections) >= 1
    assert result.timings.total_ms > 0.0


# =====================================================================
# 4. Graceful Degradation When AI Is Unavailable
# =====================================================================

def test_graceful_degradation_when_ai_unavailable():
    """Verify that if AI detector is unavailable, scan continues using deterministic detection

    and clearly exposes the AI unavailable status in errors.
    """
    mock_ai = MagicMock()
    mock_ai.is_available = False

    text = "Phone: +91 9876543210 and Aadhaar: 999912345678."

    with patch("snapshield.services.scan_service.get_default_ai_detector", return_value=mock_ai):
        result = scan_document(text, policy_mode="Personal", ai_enabled=True)

    # Deterministic detection must still succeed!
    assert len(result.detections) >= 2
    types = {d.entity_type for d in result.detections}
    assert "Phone" in types
    assert "Aadhaar" in types

    # AI unavailable status must be clearly exposed in errors
    assert any("AI detector is unavailable" in err for err in result.errors)
    assert "AI Detector Unavailable" in result.runtime_status.device_backend


def test_scan_document_ai_disabled_flag():
    """Verify ai_enabled=False bypasses AI inference phase."""
    text = "Dr. Alice Gray at 9876543210."
    result = scan_document(text, policy_mode="Personal", ai_enabled=False)

    assert result.timings.ai_inference_ms == 0.0
    # Phone still detected deterministically
    types = {d.entity_type for d in result.detections}
    assert "Phone" in types


# =====================================================================
# 5. ScanService Class Wrapper
# =====================================================================

def test_scan_service_class_wrapper():
    """Verify ScanService class wrapper works cleanly."""
    service = ScanService(default_policy_mode="Personal")
    result = service.scan("Phone: +91 9876543210")

    assert isinstance(result, ScanResult)
    assert len(result.detections) == 1
    assert result.detections[0].entity_type == "Phone"


# =====================================================================
# 6. No UI Code Invariance
# =====================================================================

def test_scan_service_has_no_streamlit_or_ui_imports():
    """Verify that scan_service has zero UI / Streamlit dependencies."""
    import sys
    import snapshield.services.scan_service as ss

    # Check imported modules in scan_service
    source_code = Path(ss.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source_code
    assert "from streamlit" not in source_code
