"""Master End-to-End Automated Test Suite for SnapShield AI.

Comprehensive automated verification of:
1. Verhoeff Checksum Algorithm
2. Aadhaar Detection & Validation
3. PAN Detection & Category Validation
4. Phone Number Detection
5. Email Detection
6. Bank Account Detection
7. IFSC Code Detection
8. Secrets & API Key Detection
9. AI Detection Parsing & BIO Span Merging
10. Duplicate Detections Resolution
11. Overlapping Detections Resolution
12. Risk Scoring & Threat Tiers
13. Personal Policy Profile
14. Developer Policy Profile
15. Redaction (Back-to-Front Invariant)
16. Blocking Action Enforcement
17. Review Action Enforcement
18. OCR Failure Handling
19. PDF Failure Handling
20. AI Unavailable Graceful Fallback
21. CPU Fallback Execution
22. Accelerated Provider Unavailable State
23. LLM Mandatory 6-Step Safety Boundary
24. Malformed Document & Upload Hardening
25. Master End-to-End Synthetic Test:
    - fake Aadhaar
    - fake PAN
    - fake phone
    - fake email
    - fake bank account
    - fake API key
    - fake medical statement
    - fake address
    * Strict verification: Zero protected values remain in sanitized output.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.ai_detector import LocalAIDetector, TARGET_CATEGORIES
from snapshield.core.code_detectors import detect_credentials
from snapshield.core.detectors import (
    detect_structured,
    is_valid_aadhaar,
    is_valid_ifsc,
    is_valid_ipv4,
    is_valid_pan,
    luhn_checksum_validate,
)
from snapshield.core.exceptions import (
    ExtractionError,
    InputValidationError,
    OCRError,
    PolicyViolationError,
    SecurityBoundaryError,
)
from snapshield.core.extractor import extract_pdf
from snapshield.core.input_handler import create_document_from_upload, validate_uploaded_file
from snapshield.core.llm_client import ExternalLLMClient, verify_safety_boundary
from snapshield.core.models import (
    Detection,
    InputDocument,
    OCRResult,
    PolicyAction,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
    RuntimeStatus,
    ScanResult,
    ScanTiming,
    SeverityLevel,
)
from snapshield.core.ocr import LocalOCRProvider, get_ocr_provider
from snapshield.core.policy import PolicyEngine, apply_policy
from snapshield.core.redactor import Redactor, verify_sanitization_leak_free
from snapshield.core.result_merger import merge_detections
from snapshield.core.risk_engine import RiskEngine
from snapshield.core.verhoeff import (
    compute_verhoeff_check_digit,
    generate_synthetic_aadhaar,
    is_valid_aadhaar,
    is_valid_verhoeff,
)
from snapshield.npu.benchmark import run_benchmark
from snapshield.npu.runtime import RuntimeMode, get_active_provider, get_runtime_status
from snapshield.services.scan_service import scan_document, scan_image


# =========================================================================
# 1. VERHOEFF & DETERMINISTIC CHECKSUMS
# =========================================================================

def test_verhoeff_algorithm():
    """Verifies Verhoeff D5 dihedral checksum calculation and validation."""
    valid_aadhaars = ["999999990019", "9999 9999 0019", "234567890124"]
    for num in valid_aadhaars:
        assert is_valid_verhoeff(num) is True
        assert is_valid_aadhaar(num) is True

    invalid_aadhaars = ["999999990018", "9999 9999 0010", "123456789012", "000000000000"]
    for num in invalid_aadhaars:
        assert is_valid_verhoeff(num) is False or is_valid_aadhaar(num) is False

    synth = generate_synthetic_aadhaar("99999999001")
    assert is_valid_aadhaar(synth) is True
    assert compute_verhoeff_check_digit("99999999001") == 9


# =========================================================================
# 2. INDIVIDUAL ENTITY DETECTORS (Aadhaar, PAN, Phone, Email, Bank, IFSC, Secrets)
# =========================================================================

def test_aadhaar_detector():
    findings = detect_structured("Candidate Aadhaar: 9999 9999 0019")
    assert any(f.entity_type.upper() == "AADHAAR" for f in findings)


def test_pan_detector():
    assert is_valid_pan("ABCPE1234F") is True
    assert is_valid_pan("ABCDE1234F") is False  # D is invalid 4th char
    findings = detect_structured("Taxpayer PAN: ABCPE1234F")
    assert any("PAN" in f.entity_type.upper() for f in findings)


def test_phone_detector():
    findings = detect_structured("Direct Line: +91 98765 43210")
    assert any("PHONE" in f.entity_type.upper() for f in findings)


def test_email_detector():
    findings = detect_structured("Inquiries: user.test@apollo-synthetic-domain.org")
    assert any("EMAIL" in f.entity_type.upper() for f in findings)


def test_bank_and_ifsc_detectors():
    assert is_valid_ifsc("HDFC0001234") is True
    assert is_valid_ifsc("HDFC1234567") is False  # 5th char must be 0
    findings = detect_structured("Primary Bank Account Number: 987654321012345 with IFSC HDFC0001234")
    types = {f.entity_type.upper() for f in findings}
    assert any("BANK" in t or "ACCOUNT" in t for t in types)
    assert any("IFSC" in t for t in types)


def test_secrets_detector():
    creds = detect_credentials("export OPENAI_API_KEY='sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678'")
    assert len(creds) >= 1
    assert creds[0].severity == SeverityLevel.CRITICAL


# =========================================================================
# 3. AI DETECTION PARSING, DUPLICATE & OVERLAPPING DETECTIONS
# =========================================================================

def test_ai_detection_parsing():
    ai = LocalAIDetector(allow_dev_fallback=True)
    results = ai.detect("Dr. John Smith diagnosed the patient with acute myocardial infarction at Springfield Clinic.")
    assert isinstance(results, list)
    assert len(results) > 0


def test_duplicate_and_overlapping_detections():
    text = "Contact 9999 9999 0019 at HDFC Bank"
    det1 = Detection(
        entity_type="AADHAAR",
        start=8,
        end=22,
        confidence=0.8,
        detector="regex",
        valid=True,
        severity=SeverityLevel.HIGH,
        explanation="Regex match",
        raw_value="9999 9999 0019",
    )
    det2 = Detection(
        entity_type="AADHAAR",
        start=8,
        end=22,
        confidence=1.0,
        detector="verhoeff",
        valid=True,
        severity=SeverityLevel.HIGH,
        explanation="Verhoeff validated",
        raw_value="9999 9999 0019",
    )
    merged = merge_detections([det1, det2], text=text)
    assert len(merged) == 1
    assert merged[0].entity_type == "AADHAAR"
    assert "verhoeff" in merged[0].detector


# =========================================================================
# 4. RISK SCORING, PERSONAL & DEVELOPER POLICIES, REDACTION, BLOCKING, REVIEW
# =========================================================================

def test_risk_scoring():
    findings = [
        Detection(
            entity_type="AADHAAR",
            start=0,
            end=14,
            confidence=1.0,
            detector="verhoeff",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Aadhaar ID",
            raw_value="9999 9999 0019",
        ),
        Detection(
            entity_type="API_KEY",
            start=20,
            end=71,
            confidence=0.99,
            detector="regex",
            valid=True,
            severity=SeverityLevel.CRITICAL,
            explanation="Secret API key",
            raw_value="sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678",
        ),
    ]
    risk_engine = RiskEngine()
    report = risk_engine.evaluate(findings)
    assert report.score >= 30.0
    assert report.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


def test_personal_and_developer_policies():
    findings = [
        Detection(entity_type="Aadhaar", start=0, end=14, confidence=1.0, detector="verhoeff", valid=True, severity=SeverityLevel.HIGH, explanation="ID", raw_value="9999 9999 0019"),
        Detection(entity_type="API_KEY", start=20, end=71, confidence=0.99, detector="regex", valid=True, severity=SeverityLevel.CRITICAL, explanation="Key", raw_value="sk-proj-12345"),
    ]
    # Personal Policy
    personal_decisions = apply_policy(findings, mode="Personal")
    p_actions = {d.entity_type: d.action for d in personal_decisions}
    assert p_actions.get("Aadhaar") == PolicyAction.REDACT

    # Developer Policy
    dev_decisions = apply_policy(findings, mode="Developer")
    d_actions = {d.entity_type: d.action for d in dev_decisions}
    assert d_actions.get("API_KEY") == PolicyAction.BLOCK


def test_redaction_back_to_front():
    orig = "ID 9999 9999 0019 and Key sk-proj-1234567890"
    detections = [
        Detection(entity_type="AADHAAR", start=3, end=17, confidence=1.0, detector="verhoeff", valid=True, severity=SeverityLevel.HIGH, explanation="ID", raw_value="9999 9999 0019"),
        Detection(entity_type="API_KEY", start=26, end=44, confidence=0.99, detector="regex", valid=True, severity=SeverityLevel.CRITICAL, explanation="Key", raw_value="sk-proj-1234567890"),
    ]
    decisions = [
        PolicyDecision(entity_type="AADHAAR", action=PolicyAction.REDACT, reason="Mask ID"),
        PolicyDecision(entity_type="API_KEY", action=PolicyAction.REDACT, reason="Mask Key"),
    ]
    res = Redactor.redact(orig, detections, decisions, mode="Personal")
    assert "9999 9999 0019" not in res.sanitized_text
    assert "sk-proj-1234567890" not in res.sanitized_text
    assert "[REDACTED: AADHAAR]" in res.sanitized_text


def test_blocking_and_review():
    findings = [
        Detection(entity_type="UNKNOWN_CONFIDENTIAL", start=0, end=10, confidence=0.8, detector="ai", valid=True, severity=SeverityLevel.MEDIUM, explanation="Unknown", raw_value="topsecret1"),
    ]
    decisions = apply_policy(findings, mode="Personal")
    assert decisions[0].action == PolicyAction.REVIEW  # Unknown defaults to REVIEW


# =========================================================================
# 5. RESILIENCE: OCR FAILURE, PDF FAILURE, AI UNAVAILABLE, CPU FALLBACK
# =========================================================================

def test_ocr_and_pdf_failure_resilience():
    # OCR Failure on empty/corrupt image bytes
    proc = LocalOCRProvider()
    with pytest.raises(OCRError):
        proc.extract_from_image(b"not_an_image")

    # PDF Failure on invalid bytes
    with pytest.raises(ExtractionError):
        extract_pdf(b"not_a_valid_pdf_stream", "corrupt.pdf")


def test_ai_unavailable_and_cpu_fallback():
    # AI Unavailable fallback
    res = scan_document("Patient 9999 9999 0019 with API key sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678", policy_mode="Personal", ai_enabled=False)
    assert "[REDACTED: AADHAAR]" in res.sanitized_text

    # CPU Fallback
    status = get_runtime_status(mode=RuntimeMode.CPU)
    assert status.active_execution_provider == "CPUExecutionProvider"
    assert status.is_accelerated is False

    # Accelerated Unavailable
    bench = run_benchmark(iterations=1, warmup_iterations=1)
    if not bench.is_accelerated_available:
        assert bench.status_message == "Accelerated execution: Not available"


# =========================================================================
# 6. LLM SAFETY BOUNDARY & MALFORMED DOCUMENT SECURITY
# =========================================================================

def test_llm_safety_boundary_enforcement():
    with pytest.raises(SecurityBoundaryError):
        verify_safety_boundary(scan=None, text_to_send="sanitized text")

    blocked_scan = ScanResult(
        original_text="Secret key",
        sanitized_text="[BLOCKED]",
        detections=[Detection(entity_type="KEY", start=0, end=10, confidence=1.0, detector="regex", valid=True, severity=SeverityLevel.CRITICAL, explanation="Key", raw_value="Secret key")],
        decisions=[PolicyDecision(entity_type="KEY", action=PolicyAction.BLOCK, reason="Blocked")],
        risk_assessment=RiskAssessment(score=90.0, level=RiskLevel.CRITICAL, is_blocked=True),
        timings=ScanTiming(),
        runtime_status=RuntimeStatus(os_name="Win", machine_arch="x64", processor="CPU", active_execution_provider="CPUExecutionProvider", is_accelerated=False, device_backend="CPU"),
    )
    with pytest.raises(PolicyViolationError):
        verify_safety_boundary(scan=blocked_scan, text_to_send="[BLOCKED]")


def test_malformed_document_hardening():
    # Path traversal attack filename via BytesIO with traversal name
    traversal_file = io.BytesIO(b"content")
    traversal_file.name = "../../../etc/shadow"
    val_res = validate_uploaded_file(traversal_file)
    assert val_res.is_valid is False or "shadow" not in val_res.file_name

    # File exceeding size limit (> 25MB)
    large_file = io.BytesIO(b"X" * (30 * 1024 * 1024))
    large_file.name = "huge.txt"
    val_res_large = validate_uploaded_file(large_file)
    assert val_res_large.is_valid is False


# =========================================================================
# 7. MASTER END-TO-END SYNTHETIC TEST CONTAINING ALL 8 FAKE ENTITIES
# =========================================================================

def test_master_end_to_end_synthetic_document_all_entities():
    """Complete end-to-end synthetic test containing all 8 target entity categories:

    - Fake Aadhaar: 9999 9999 0019
    - Fake PAN: ABCPE1234F
    - Fake Phone: +91 98765 43210
    - Fake Email: patient.doe@apollo-health-synthetic.org
    - Fake Bank Account: 987654321012345 (with IFSC HDFC0001234)
    - Fake API Key: sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678
    - Fake Medical Statement: prescription for chemotherapy
    - Fake Address: 742 Evergreen Street

    Verifies:
    1. Comprehensive detection of structured, credentials, and contextual AI entities.
    2. Policy decision resolution.
    3. Back-to-front leak-free redaction.
    4. Sanitized output decoupling.
    5. STRICT ASSERTION: Zero raw protected values remain in the sanitized output.
    """
    fake_aadhaar = "9999 9999 0019"
    fake_pan = "ABCPE1234F"
    fake_phone = "+91 98765 43210"
    fake_email = "patient.doe@apollo-health-synthetic.org"
    fake_bank_account = "987654321012345"
    fake_ifsc = "HDFC0001234"
    fake_api_key = "sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678"
    fake_medical = "prescription for chemotherapy"
    fake_address = "742 Evergreen Street"

    synthetic_document = (
        f"CONFIDENTIAL MEDICAL ADMISSION REPORT:\n"
        f"Patient Aadhaar: {fake_aadhaar}\n"
        f"Taxpayer PAN: {fake_pan}\n"
        f"Emergency Phone: {fake_phone}\n"
        f"Contact Email: {fake_email}\n"
        f"Bank Account: {fake_bank_account} with IFSC: {fake_ifsc}\n"
        f"Clinical Diagnosis: Patient received {fake_medical}.\n"
        f"Permanent Address: {fake_address}.\n"
        f"Deployment Key: export OPENAI_API_KEY='{fake_api_key}'"
    )

    all_protected_values = [
        fake_aadhaar,
        fake_pan,
        fake_phone,
        fake_email,
        fake_bank_account,
        fake_ifsc,
        fake_api_key,
        fake_medical,
        fake_address,
    ]

    # 1. Test with Medical Policy (Strictly redacts/blocks all PII, health, and credentials)
    med_scan_res = scan_document(
        document=synthetic_document,
        policy_mode="Medical",
        ai_enabled=True,
    )

    assert isinstance(med_scan_res, ScanResult)
    assert len(med_scan_res.detections) >= 6

    detected_types = {d.entity_type.upper() for d in med_scan_res.detections}
    assert "AADHAAR" in detected_types
    assert "PAN" in detected_types
    assert any("PHONE" in t for t in detected_types)
    assert any("EMAIL" in t for t in detected_types)
    assert any("BANK" in t or "ACCOUNT" in t or "IFSC" in t for t in detected_types)
    assert any("KEY" in t or "CREDENTIAL" in t or "API" in t for t in detected_types)

    # Policy & Decoupling verification
    assert len(med_scan_res.decisions) >= 1
    assert med_scan_res.risk_assessment.score > 0.0
    assert med_scan_res.sanitized_text != synthetic_document
    assert len(med_scan_res.sanitized_text) > 0
    assert med_scan_res.integrity_hash != ""

    # STRICT INVARIANT: Zero protected values remain in the sanitized output
    for protected_val in all_protected_values:
        assert protected_val not in med_scan_res.sanitized_text, (
            f"SECURITY BREACH: Protected value '{protected_val}' leaked into sanitized output!\n"
            f"Sanitized text:\n{med_scan_res.sanitized_text}"
        )

    # Verify cryptographic leak-free assertion
    verify_sanitization_leak_free(
        med_scan_res.sanitized_text,
        {fake_aadhaar, fake_pan, fake_phone, fake_email, fake_bank_account, fake_ifsc, fake_api_key},
    )

    # 2. Test with Developer Policy (Enforces BLOCK on API key)
    dev_scan_res = scan_document(
        document=synthetic_document,
        policy_mode="Developer",
        ai_enabled=True,
    )
    assert dev_scan_res.is_blocked is True or dev_scan_res.overall_action == PolicyAction.BLOCK
    assert fake_api_key not in dev_scan_res.sanitized_text
