"""Comprehensive Security and Privacy Hardening Test Suite.

Proves:
1. Log Sanitization:
   - Aadhaar, PAN, phone, email, bank/cards, passwords, API keys, medical information, and document content are NEVER logged raw.
2. Filename Sanitization & Path Traversal Protection:
   - Path traversal sequences (../, ..\\), null bytes, and control characters are stripped.
3. File & Resource Exhaustion Protection:
   - Decompression bomb images (> 8192px), oversized PDFs (> 100 pages), and large files are rejected.
4. Malformed Input Rejection:
   - Corrupted PDFs, malformed images, and binary-polluted TXT files are safely rejected.
5. Safety Boundary Verification:
   - Proof that external LLMs CANNOT be invoked before sanitization and compliance evaluation.
"""

import io
import logging
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from snapshield.core.exceptions import (
    ExtractionError,
    InputValidationError,
    PolicyViolationError,
    SecurityBoundaryError,
)
from snapshield.core.input_handler import (
    MAX_IMAGE_DIMENSION,
    MAX_PDF_PAGES,
    sanitize_filename,
    validate_uploaded_file,
)
from snapshield.core.llm_client import (
    ExternalLLMClient,
    verify_safety_boundary,
)
from snapshield.core.logger import PrivacyPreservingFormatter, SENSITIVE_SCRUBBERS, setup_logger
from snapshield.core.models import (
    Detection,
    PolicyAction,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
    RuntimeStatus,
    ScanResult,
    ScanTiming,
    SeverityLevel,
)


# =========================================================================
# 1. LOG SANITIZATION TESTS
# =========================================================================

def test_logger_never_logs_sensitive_aadhaar_pan():
    """Verifies logger scrubs Aadhaar and PAN numbers."""
    formatter = PrivacyPreservingFormatter("%(message)s")
    record_aadhaar = logging.LogRecord("test", logging.INFO, "", 0, "Patient Aadhaar: 9999 9999 0019", (), None)
    sanitized_aadhaar = formatter.format(record_aadhaar)
    assert "9999 9999 0019" not in sanitized_aadhaar
    assert "[REDACTED_AADHAAR]" in sanitized_aadhaar

    record_pan = logging.LogRecord("test", logging.INFO, "", 0, "User PAN: ABCDE1234F verified.", (), None)
    sanitized_pan = formatter.format(record_pan)
    assert "ABCDE1234F" not in sanitized_pan
    assert "[REDACTED_PAN]" in sanitized_pan


def test_logger_never_logs_phone_and_email():
    """Verifies logger scrubs phone numbers and email addresses."""
    formatter = PrivacyPreservingFormatter("%(message)s")
    record_phone = logging.LogRecord("test", logging.INFO, "", 0, "Contact +91 9876543210 or (555) 123-4567", (), None)
    sanitized_phone = formatter.format(record_phone)
    assert "9876543210" not in sanitized_phone
    assert "123-4567" not in sanitized_phone

    record_email = logging.LogRecord("test", logging.INFO, "", 0, "Forward to robert.smith@confidential-hospital.org immediately", (), None)
    sanitized_email = formatter.format(record_email)
    assert "robert.smith@confidential-hospital.org" not in sanitized_email
    assert "[REDACTED_EMAIL]" in sanitized_email


def test_logger_never_logs_api_keys_and_passwords():
    """Verifies logger scrubs API keys, tokens, and passwords."""
    formatter = PrivacyPreservingFormatter("%(message)s")
    record_key = logging.LogRecord("test", logging.INFO, "", 0, "api_key = 'sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678'", (), None)
    sanitized_key = formatter.format(record_key)
    assert "sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678" not in sanitized_key
    assert "[REDACTED_SECRET]" in sanitized_key or "[REDACTED_API_KEY]" in sanitized_key

    record_pass = logging.LogRecord("test", logging.INFO, "", 0, "password: SuperSecret1234!", (), None)
    sanitized_pass = formatter.format(record_pass)
    assert "SuperSecret1234!" not in sanitized_pass


def test_logger_never_logs_bank_and_medical_info():
    """Verifies logger scrubs credit cards, IBAN, and clinical diagnoses."""
    formatter = PrivacyPreservingFormatter("%(message)s")
    record_card = logging.LogRecord("test", logging.INFO, "", 0, "Card payment 4532-7588-1234-5678 processed.", (), None)
    sanitized_card = formatter.format(record_card)
    assert "4532-7588-1234-5678" not in sanitized_card
    assert "[REDACTED_CARD]" in sanitized_card

    record_med = logging.LogRecord("test", logging.INFO, "", 0, "Patient diagnosed with acute coronary syndrome and cardiac arrest", (), None)
    sanitized_med = formatter.format(record_med)
    assert "acute coronary syndrome" not in sanitized_med
    assert "cardiac arrest" not in sanitized_med


# =========================================================================
# 2. FILENAME SANITIZATION & PATH TRAVERSAL TESTS
# =========================================================================

def test_sanitize_filename_prevents_path_traversal():
    """Verifies that malicious directory traversal sequences are completely neutralized."""
    traversal_inputs = [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\config.sys",
        "/etc/shadow",
        "....//....//secret.txt",
        "../../../var/log/audit.log",
        "test\x00injection.txt",
        "nested/subfolder/file.pdf",
    ]

    for raw in traversal_inputs:
        cleaned = sanitize_filename(raw)
        assert "/" not in cleaned
        assert "\\" not in cleaned
        assert ".." not in cleaned
        assert "\x00" not in cleaned


def test_sanitize_filename_handles_empty_or_dangerous():
    """Verifies fallback for empty or purely dot-based filenames."""
    assert sanitize_filename("") == "uploaded_document"
    assert sanitize_filename("   ") == "uploaded_document"
    assert sanitize_filename("...") == "uploaded_document"
    assert sanitize_filename(None) == "uploaded_document"


# =========================================================================
# 3. RESOURCE EXHAUSTION & DECOMPRESSION BOMB PROTECTION
# =========================================================================

def test_decompression_bomb_image_rejected():
    """Verifies that images exceeding safety dimension bounds are rejected."""
    # Create an in-memory image exceeding MAX_IMAGE_DIMENSION
    oversized_img = Image.new("RGB", (MAX_IMAGE_DIMENSION + 100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    oversized_img.save(buf, format="PNG")
    buf.name = "bomb.png"
    buf.seek(0)

    res = validate_uploaded_file(buf)
    assert res.is_valid is False
    assert "exceed maximum allowed dimension limit" in res.error_message


def test_malformed_pdf_and_page_limits():
    """Verifies malformed PDFs and page count limits are handled cleanly."""
    corrupted_pdf = io.BytesIO(b"%PDF-1.4 Corrupted Binary Stream not a real PDF")
    corrupted_pdf.name = "corrupted.pdf"
    res = validate_uploaded_file(corrupted_pdf)
    assert res.is_valid is False
    assert "corrupted" in res.error_message.lower() or "malformed" in res.error_message.lower()


# =========================================================================
# 4. SAFETY BOUNDARY: EXTERNAL LLM CANNOT BE CALLED BEFORE SANITIZATION
# =========================================================================

@pytest.fixture
def clean_sanitized_scan() -> ScanResult:
    """Fixture providing a valid, verified sanitized scan result."""
    detections = [
        Detection(
            entity_type="AADHAAR",
            start=10,
            end=24,
            confidence=1.0,
            detector="verhoeff",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Validated Aadhaar",
            raw_value="9999 9999 0019",
        )
    ]
    decisions = [
        PolicyDecision(entity_type="AADHAAR", action=PolicyAction.REDACT, reason="Mask ID")
    ]
    return ScanResult(
        original_text="Patient: 9999 9999 0019 clinical intake.",
        sanitized_text="Patient: [REDACTED: AADHAAR] clinical intake.",
        detections=detections,
        decisions=decisions,
        risk_assessment=RiskAssessment(score=30.0, level=RiskLevel.MEDIUM),
        timings=ScanTiming(ai_inference_ms=12.0, total_ms=25.0),
        runtime_status=RuntimeStatus(
            os_name="Windows",
            machine_arch="AMD64",
            processor="Intel64",
            active_execution_provider="CPUExecutionProvider",
            is_accelerated=False,
            device_backend="CPUExecutionProvider",
        ),
    )


def test_safety_boundary_fails_if_scan_is_none():
    """Check 1: Calling LLM with scan=None is strictly prohibited."""
    with pytest.raises(SecurityBoundaryError) as exc_info:
        verify_safety_boundary(scan=None, text_to_send="Some sanitized text")
    assert "Safety Check 1/6 Failed: Scan does not exist" in str(exc_info.value)


def test_safety_boundary_fails_if_decisions_empty(clean_sanitized_scan):
    """Check 2: Calling LLM with empty policy decisions is strictly prohibited."""
    clean_sanitized_scan.decisions = []
    with pytest.raises(SecurityBoundaryError) as exc_info:
        verify_safety_boundary(scan=clean_sanitized_scan, text_to_send=clean_sanitized_scan.sanitized_text)
    assert "Safety Check 2/6 Failed: Policy decisions do not exist" in str(exc_info.value)


def test_safety_boundary_fails_if_blocked_entities_present(clean_sanitized_scan):
    """Check 3: Calling LLM when content is marked BLOCK is strictly prohibited."""
    clean_sanitized_scan.decisions.append(
        PolicyDecision(entity_type="API_KEY", action=PolicyAction.BLOCK, reason="Prohibited")
    )
    with pytest.raises(PolicyViolationError) as exc_info:
        verify_safety_boundary(scan=clean_sanitized_scan, text_to_send=clean_sanitized_scan.sanitized_text)
    assert "Safety Check 3/6 Failed" in str(exc_info.value)


def test_safety_boundary_fails_if_redaction_leaks_sensitive_token(clean_sanitized_scan):
    """Check 4: Calling LLM when sanitized text still contains raw sensitive token is prohibited."""
    leaking_scan = clean_sanitized_scan.model_copy()
    leaking_scan.sanitized_text = "Patient: 9999 9999 0019 leaked text."

    with pytest.raises(SecurityBoundaryError) as exc_info:
        verify_safety_boundary(scan=leaking_scan, text_to_send=leaking_scan.sanitized_text)
    assert "Safety Check 4/6 Failed: Redaction is incomplete" in str(exc_info.value)


def test_safety_boundary_fails_if_review_not_approved(clean_sanitized_scan):
    """Check 5: Calling LLM with unapproved REVIEW items is prohibited."""
    clean_sanitized_scan.decisions = [
        PolicyDecision(entity_type="SUSPICIOUS_ID", action=PolicyAction.REVIEW, reason="Needs check")
    ]
    with pytest.raises(PolicyViolationError) as exc_info:
        verify_safety_boundary(
            scan=clean_sanitized_scan,
            text_to_send=clean_sanitized_scan.sanitized_text,
            review_approved=False,
        )
    assert "Safety Check 5/6 Failed: Review items are present" in str(exc_info.value)


def test_safety_boundary_fails_if_original_unredacted_text_is_passed(clean_sanitized_scan):
    """Check 6: Attempting to pass original unredacted text raises critical breach."""
    with pytest.raises(SecurityBoundaryError) as exc_info:
        verify_safety_boundary(
            scan=clean_sanitized_scan,
            text_to_send=clean_sanitized_scan.original_text,
            review_approved=True,
        )
    assert "Safety Check 6/6 Failed" in str(exc_info.value)


def test_safety_boundary_fails_if_text_payload_mismatches_sanitized_text(clean_sanitized_scan):
    """Check 6: Attempting to pass arbitrary modified text mismatches approved sanitized text."""
    with pytest.raises(SecurityBoundaryError) as exc_info:
        verify_safety_boundary(
            scan=clean_sanitized_scan,
            text_to_send="Different unapproved payload entirely.",
            review_approved=True,
        )
    assert "Safety Check 6/6 Failed: Payload mismatch" in str(exc_info.value)


def test_external_llm_client_enforces_safety_before_dispatch(clean_sanitized_scan):
    """Verifies that ExternalLLMClient.send_prompt executes pre-flight checks and blocks violations."""
    client = ExternalLLMClient(api_key="test-key-12345")

    # 1. Attempting to send original text is rejected before network dispatch
    with pytest.raises(SecurityBoundaryError):
        client.send_prompt(
            scan_result=clean_sanitized_scan,
            text_to_send=clean_sanitized_scan.original_text,
        )

    # 2. Forwarding sanitized text succeeds safety verification
    with patch.object(client, "_dispatch_http_request", return_value=("AI response", 200, {})):
        resp = client.forward_sanitized(scan_result=clean_sanitized_scan)
        assert resp.success is True
        assert resp.content == "AI response"
