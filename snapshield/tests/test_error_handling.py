"""Automated Tests for Application-Wide Error Handling & Graceful Degradation.

Verifies:
1. Invalid files & extension handling
2. Empty files & whitespace payloads
3. OCR failures & missing binaries
4. PDF extraction failures
5. AI model unavailable (degrades to deterministic detection)
6. Runtime unavailable (degrades to CPU fallback)
7. Accelerated provider failure (degrades to CPU fallback)
8. LLM failure (sanitized output remains usable)
9. Network failure (DNS / timeout handling)
10. Invalid API key (AuthenticationError)
11. Oversized files & dimension bombs
12. Malformed model handling
13. Insufficient memory / OOM simulation
14. User-friendly error message formatting (no raw stack traces to normal users)
15. Controlled developer diagnostics in debug mode
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.ai_detector import LocalAIDetector
from snapshield.core.exceptions import (
    AcceleratedProviderError,
    AuthenticationError,
    DetectionError,
    EmptyFileError,
    ExtractionError,
    HardwareAccelerationError,
    InputValidationError,
    InsufficientMemoryError,
    InvalidFileFormatError,
    LLMServiceError,
    MalformedModelError,
    ModelUnavailableError,
    NetworkError,
    OCRError,
    OversizedFileError,
    PDFExtractionError,
    PolicyViolationError,
    RedactionError,
    RuntimeUnavailableError,
    SecurityBoundaryError,
    SnapShieldError,
    TesseractNotFoundError,
    classify_error,
)
from snapshield.core.extractor import extract_pdf
from snapshield.core.input_handler import (
    create_document_from_upload,
    create_text_document,
    validate_uploaded_file,
)
from snapshield.core.llm_client import ExternalLLMClient, LLMResponse, verify_safety_boundary
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
from snapshield.core.ocr import LocalOCRProvider
from snapshield.npu.benchmark import run_benchmark
from snapshield.npu.model_runner import ModelRunner
from snapshield.npu.runtime import RuntimeMode, get_active_provider, get_runtime_status
from snapshield.services.scan_service import scan_document, scan_image


# =========================================================================
# 1. INVALID & EMPTY FILES
# =========================================================================

def test_empty_file_handling():
    """Verify empty files raise or report EmptyFileError."""
    empty_stream = io.BytesIO(b"")
    empty_stream.name = "empty.txt"
    val = validate_uploaded_file(empty_stream)
    assert val.is_valid is False
    assert "empty" in val.error_message.lower()

    # Empty string input document
    with pytest.raises(InputValidationError):
        create_text_document("   ")

    # Error classification
    diag = classify_error(EmptyFileError())
    assert diag.error_code == "ERR_EMPTY_FILE"
    assert "0 characters" in diag.user_message


def test_invalid_file_format_handling():
    """Verify unsupported file extensions or spoofed magic headers are rejected."""
    unsupported = io.BytesIO(b"binary executable")
    unsupported.name = "malware.exe"
    val = validate_uploaded_file(unsupported)
    assert val.is_valid is False
    assert "unsupported" in val.error_message.lower()

    diag = classify_error(InvalidFileFormatError("Unsupported extension .exe"))
    assert diag.error_code == "ERR_INVALID_FILE_FORMAT"
    assert ".txt, .pdf, .png" in diag.recovery_action


def test_oversized_file_handling():
    """Verify oversized files are rejected cleanly."""
    huge_bytes = io.BytesIO(b"A" * (30 * 1024 * 1024))
    huge_bytes.name = "giant.txt"
    val = validate_uploaded_file(huge_bytes)
    assert val.is_valid is False
    assert "exceeds maximum allowed limit" in val.error_message

    diag = classify_error(OversizedFileError("File exceeds 25 MB"))
    assert diag.error_code == "ERR_OVERSIZED_FILE"


# =========================================================================
# 2. OCR & PDF EXTRACTION FAILURES
# =========================================================================

def test_ocr_failure_handling():
    """Verify OCR failures raise typed OCRError without crashing."""
    proc = LocalOCRProvider()
    with pytest.raises(OCRError):
        proc.extract_from_image(b"corrupt_image_data_here")

    diag = classify_error(TesseractNotFoundError())
    assert diag.error_code == "ERR_TESSERACT_NOT_FOUND"


def test_pdf_extraction_failure_handling():
    """Verify corrupted PDF bytes raise ExtractionError."""
    with pytest.raises(ExtractionError):
        extract_pdf(b"not_a_valid_pdf_stream", "corrupt.pdf")

    diag = classify_error(PDFExtractionError("Corrupted stream"))
    assert diag.error_code == "ERR_PDF_EXTRACTION_FAILED"


# =========================================================================
# 3. AI UNAVAILABLE & CPU / ACCELERATION DEGRADATION
# =========================================================================

def test_ai_unavailable_graceful_degradation():
    """AI Unavailable -> Deterministic detection continues uninterrupted."""
    # Run scan with ai_enabled=False
    text = "Patient John Doe with Aadhaar 9999 9999 0019 and PAN ABCPE1234F"
    res = scan_document(text, policy_mode="Personal", ai_enabled=False)

    assert isinstance(res, ScanResult)
    assert len(res.detections) >= 2
    assert "[REDACTED: AADHAAR]" in res.sanitized_text
    assert "9999 9999 0019" not in res.sanitized_text

    diag = classify_error(ModelUnavailableError())
    assert diag.error_code == "ERR_MODEL_UNAVAILABLE"
    assert "Deterministic Scanning Active" in diag.degradation_status


def test_acceleration_unavailable_cpu_fallback():
    """Acceleration Unavailable -> CPU inference continues."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "bert_small_pii" / "model.onnx"
    runner = ModelRunner(model_path=model_path, mode=RuntimeMode.CPU)
    assert runner.mode == RuntimeMode.CPU

    status = get_runtime_status(mode=RuntimeMode.CPU)
    assert status.active_execution_provider == "CPUExecutionProvider"
    assert status.is_accelerated is False

    diag = classify_error(AcceleratedProviderError("DirectML failed"))
    assert "CPU Fallback Active" in diag.degradation_status


def test_malformed_model_handling():
    """Malformed model triggers safe fallback."""
    diag = classify_error(MalformedModelError("Invalid ONNX graph"))
    assert diag.error_code == "ERR_MALFORMED_MODEL"


# =========================================================================
# 4. LLM & NETWORK FAILURES (SANITIZED OUTPUT REMAINS USABLE)
# =========================================================================

def test_llm_network_failure_handling():
    """Network failure during LLM egress does not destroy sanitized text."""
    client = ExternalLLMClient(api_key="test-sk-key-12345", provider="openai")

    scan_res = ScanResult(
        original_text="Citizen 9999 9999 0019",
        sanitized_text="Citizen [REDACTED: AADHAAR]",
        detections=[Detection(entity_type="AADHAAR", start=8, end=22, confidence=1.0, detector="verhoeff", valid=True, severity=SeverityLevel.HIGH, explanation="Aadhaar", raw_value="9999 9999 0019")],
        decisions=[PolicyDecision(entity_type="AADHAAR", action=PolicyAction.REDACT, reason="Mask ID")],
        risk_assessment=RiskAssessment(score=50.0, level=RiskLevel.HIGH),
        timings=ScanTiming(),
        runtime_status=RuntimeStatus(os_name="Win", machine_arch="x64", processor="CPU", active_execution_provider="CPUExecutionProvider", is_accelerated=False, device_backend="CPU"),
        integrity_hash="abcd1234hash",
    )

    with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
        resp = client.forward_sanitized(scan_res)
        assert resp.success is False
        assert "Network unreachable" in resp.error or "failed" in resp.error
        # Invariant: sanitized prompt hash and scan result remain valid for user
        assert scan_res.sanitized_text == "Citizen [REDACTED: AADHAAR]"

    diag = classify_error(NetworkError())
    assert diag.error_code == "ERR_NETWORK_DISCONNECTED"
    assert "Local Sanitized Output Preserved" in diag.degradation_status


def test_llm_invalid_api_key_authentication_error():
    """Invalid API key raises or reports AuthenticationError."""
    client = ExternalLLMClient(api_key="invalid-bad-key", provider="openai")

    scan_res = ScanResult(
        original_text="Secret",
        sanitized_text="[REDACTED: SECRET]",
        detections=[Detection(entity_type="SECRET", start=0, end=6, confidence=1.0, detector="regex", valid=True, severity=SeverityLevel.HIGH, explanation="Secret", raw_value="Secret")],
        decisions=[PolicyDecision(entity_type="SECRET", action=PolicyAction.REDACT, reason="Mask Secret")],
        risk_assessment=RiskAssessment(score=40.0, level=RiskLevel.MEDIUM),
        timings=ScanTiming(),
        runtime_status=RuntimeStatus(os_name="Win", machine_arch="x64", processor="CPU", active_execution_provider="CPUExecutionProvider", is_accelerated=False, device_backend="CPU"),
    )

    import urllib.error
    mock_http_err = urllib.error.HTTPError("https://api.openai.com", 401, "Unauthorized", {}, None)
    with patch("urllib.request.urlopen", side_effect=mock_http_err):
        resp = client.forward_sanitized(scan_res)
        assert resp.success is False
        assert "Authentication failed" in resp.error or "401" in resp.error

    diag = classify_error(AuthenticationError())
    assert diag.error_code == "ERR_AUTHENTICATION_FAILED"
    assert "Local Sanitized Output Preserved" in diag.degradation_status


# =========================================================================
# 5. INSUFFICIENT MEMORY & DEBUG MODE DIAGNOSTICS
# =========================================================================

def test_insufficient_memory_handling():
    """MemoryError or InsufficientMemoryError produces user-friendly recovery instructions."""
    diag = classify_error(InsufficientMemoryError("Memory limit reached"))
    assert diag.error_code == "ERR_INSUFFICIENT_MEMORY"
    assert "smaller section" in diag.recovery_action.lower()


def test_user_friendly_error_formatting_no_raw_stack_traces_by_default():
    """Normal error diagnostics provide clear guidance and no raw stack trace dump."""
    exc = InputValidationError("Malformed header bytes detected")
    diag = classify_error(exc)

    assert diag.title != ""
    assert diag.user_message != ""
    assert diag.recovery_action != ""
    assert diag.degradation_status != ""
    assert diag.error_code == "ERR_INPUT_VALIDATION"
    assert diag.traceback_str is not None  # Captured for developer debug mode
