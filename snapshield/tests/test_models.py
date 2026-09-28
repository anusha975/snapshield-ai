"""Comprehensive tests for SnapShield AI strongly typed data models.

Validates:
1. Model creation and field constraints
2. Validation rules (e.g. span boundaries, confidence limits, score bounds)
3. Safe display helpers and masking functions (Rule 2: Never expose raw sensitive values)
4. Serialization and deserialization roundtrips (model_dump, model_dump_json, model_validate)
5. End-to-end ScanResult composition and integrity hashing
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from snapshield.core.models import (
    BenchmarkResult,
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
    mask_sensitive_value,
)


# ==========================================
# 1. Masking & Privacy Helper Tests
# ==========================================

def test_mask_sensitive_value_credit_card():
    """Verify payment card masking preserves only last 4 digits."""
    raw_card = "4532-0150-1234-5671"
    masked = mask_sensitive_value(raw_card, entity_type="PAYMENT_CARD")
    assert masked == "****-****-****-5671"
    assert "4532" not in masked
    assert "0150" not in masked


def test_mask_sensitive_value_government_id():
    """Verify SSN / Aadhaar / PAN masking."""
    raw_ssn = "123-45-6789"
    masked = mask_sensitive_value(raw_ssn, entity_type="GOVERNMENT_ID_SSN")
    assert masked == "***-**-6789"
    assert "123" not in masked


def test_mask_sensitive_value_email():
    """Verify email masking preserves domain while masking username."""
    raw_email = "employee.john@enterprise.org"
    masked = mask_sensitive_value(raw_email, entity_type="EMAIL")
    assert "@enterprise.org" in masked
    assert "employee.john" not in masked


def test_mask_sensitive_value_secrets():
    """Verify API keys, tokens, and secrets are thoroughly redacted."""
    raw_key = "sk-live-998877665544332211"
    masked = mask_sensitive_value(raw_key, entity_type="API_KEY")
    assert "[REDACTED_SECRET]" in masked
    assert "998877665544332211" not in masked


def test_detection_safe_helpers_never_expose_raw_values():
    """Verify Detection.__repr__ and Detection.safe_dict() never leak raw secrets."""
    detection = Detection(
        entity_type="CREDENTIAL_API_KEY",
        start=15,
        end=45,
        confidence=0.99,
        detector="regex_pattern",
        valid=True,
        severity=SeverityLevel.CRITICAL,
        explanation="Detected high-entropy API authentication token.",
        raw_value="sk-verySecretKeyThatMustNeverLeak12345",
    )

    # __repr__ must NOT contain the raw value
    repr_str = repr(detection)
    assert "sk-verySecretKey" not in repr_str
    assert "CREDENTIAL_API_KEY" in repr_str

    # safe_dict must return masked representation, NOT raw_value
    safe_data = detection.safe_dict()
    assert "raw_value" not in safe_data
    assert "sk-verySecretKey" not in json.dumps(safe_data)
    assert "[REDACTED_SECRET]" in safe_data["masked_value"]


# ==========================================
# 2. Validation Constraints Tests
# ==========================================

def test_detection_span_validation():
    """Verify that Detection rejects end offset < start offset."""
    # Valid span
    Detection(
        entity_type="PHONE",
        start=10,
        end=20,
        confidence=0.85,
        detector="regex",
        explanation="Phone match",
    )

    # Invalid span (end < start)
    with pytest.raises(ValidationError) as exc:
        Detection(
            entity_type="PHONE",
            start=20,
            end=10,
            confidence=0.85,
            detector="regex",
            explanation="Invalid span",
        )
    assert "Span end (10) cannot be strictly less than start (20)" in str(exc.value)


def test_detection_confidence_bounds():
    """Verify confidence must be strictly between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        Detection(
            entity_type="SSN",
            start=0,
            end=11,
            confidence=1.5,  # Invalid > 1.0
            detector="regex",
            explanation="Invalid confidence",
        )

    with pytest.raises(ValidationError):
        Detection(
            entity_type="SSN",
            start=0,
            end=11,
            confidence=-0.1,  # Invalid < 0.0
            detector="regex",
            explanation="Negative confidence",
        )


def test_risk_assessment_bounds():
    """Verify RiskAssessment score bounds (0.0 to 100.0)."""
    # Valid
    ra = RiskAssessment(score=75.5, level=RiskLevel.HIGH, findings_count=3)
    assert ra.score == 75.5

    # Out of bounds
    with pytest.raises(ValidationError):
        RiskAssessment(score=105.0, level=RiskLevel.CRITICAL)

    with pytest.raises(ValidationError):
        RiskAssessment(score=-1.0, level=RiskLevel.LOW)


def test_scan_timing_calculation():
    """Verify ScanTiming component sum calculation."""
    timing = ScanTiming(
        extraction_ms=12.4,
        ocr_ms=0.0,
        detection_ms=25.6,
        risk_assessment_ms=3.1,
        policy_ms=1.5,
        redaction_ms=5.4,
    )
    total = timing.calculate_total()
    assert total == pytest.approx(48.0, 0.01)
    assert timing.total_ms == pytest.approx(48.0, 0.01)


# ==========================================
# 3. Serialization and Deserialization Tests
# ==========================================

def test_input_document_serialization():
    """Verify InputDocument creation, immutability, and serialization."""
    doc = InputDocument(
        document_type="RAW_TEXT",
        raw_content="Confidential financial report content...",
        file_name="report.txt",
        file_size_bytes=1024,
    )

    # Immutability
    with pytest.raises(ValidationError):
        doc.document_type = "PDF"  # type: ignore

    # Safe summary must not leak raw_content
    summary = doc.safe_summary()
    assert "raw_content" not in summary
    assert summary["file_size_bytes"] == 1024

    # Serialization roundtrip
    dumped_json = doc.model_dump_json()
    reloaded = InputDocument.model_validate_json(dumped_json)
    assert reloaded.document_id == doc.document_id
    assert reloaded.file_name == doc.file_name


def test_ocr_result_serialization():
    """Verify OCRResult serialization and safe summary."""
    ocr = OCRResult(
        text="Extracted text from invoice screenshot",
        confidence=0.92,
        engine="tesseract_local",
        word_count=5,
        char_count=38,
        processing_time_ms=145.2,
    )

    summary = ocr.safe_summary()
    assert summary["confidence"] == 0.92
    assert summary["engine"] == "tesseract_local"

    # Json dump & reload
    dumped = ocr.model_dump()
    reloaded = OCRResult.model_validate(dumped)
    assert reloaded.confidence == 0.92


def test_policy_decision_model():
    """Verify PolicyDecision model validation and serialization."""
    decision = PolicyDecision(
        entity_type="CREDENTIAL_API_KEY",
        action=PolicyAction.BLOCK,
        reason="Developer policy forbids transmission of API keys.",
        profile_applied="Developer",
    )
    assert decision.action == PolicyAction.BLOCK

    json_str = decision.model_dump_json()
    reloaded = PolicyDecision.model_validate_json(json_str)
    assert reloaded.action == PolicyAction.BLOCK
    assert reloaded.profile_applied == "Developer"


def test_benchmark_result_model():
    """Verify BenchmarkResult serialization."""
    bm = BenchmarkResult(
        execution_provider="QNNExecutionProvider",
        sample_name="synthetic_finance_01",
        iterations=50,
        mean_latency_ms=14.2,
        p95_latency_ms=18.5,
        throughput_items_per_sec=70.4,
        hardware_details={"npu": "Qualcomm Hexagon"},
    )
    assert bm.iterations == 50

    data = bm.model_dump()
    reloaded = BenchmarkResult.model_validate(data)
    assert reloaded.mean_latency_ms == 14.2
    assert reloaded.hardware_details["npu"] == "Qualcomm Hexagon"


def test_full_scan_result_pipeline_serialization():
    """Verify complete ScanResult model with detections, decisions, risk, and integrity hash."""
    detection = Detection(
        entity_type="PAYMENT_CARD",
        start=10,
        end=29,
        confidence=1.0,
        detector="luhn_checksum",
        valid=True,
        severity=SeverityLevel.HIGH,
        explanation="Valid Luhn Mod-10 credit card number.",
        raw_value="4532-0150-1234-5671",
    )

    decision = PolicyDecision(
        entity_type="PAYMENT_CARD",
        action=PolicyAction.REDACT,
        reason="Personal policy requires masking payment card numbers.",
        profile_applied="Personal",
    )

    risk = RiskAssessment(
        score=45.0,
        level=RiskLevel.HIGH,
        findings_count=1,
        category_breakdown={"PAYMENT_CARD": 1},
        reasons=["Detected 1 payment card."],
        is_blocked=False,
    )

    timing = ScanTiming(
        extraction_ms=1.5,
        ocr_ms=0.0,
        detection_ms=3.2,
        risk_assessment_ms=0.5,
        policy_ms=0.8,
        redaction_ms=1.2,
        total_ms=7.2,
    )

    runtime = RuntimeStatus(
        os_name="Windows",
        machine_arch="AMD64",
        processor="Intel64 Family",
        active_execution_provider="CPUExecutionProvider",
        is_accelerated=False,
        device_backend="Verified CPU Fallback",
    )

    scan_res = ScanResult(
        original_text="Paid using 4532-0150-1234-5671 on Friday.",
        sanitized_text="Paid using [REDACTED_PAYMENT_CARD] on Friday.",
        detections=[detection],
        decisions=[decision],
        risk_assessment=risk,
        timings=timing,
        runtime_status=runtime,
        errors=[],
    )

    # Post init must have generated non-empty SHA-256 integrity hash
    assert scan_res.integrity_hash != ""
    assert len(scan_res.integrity_hash) == 64
    assert scan_res.is_blocked is False
    assert scan_res.overall_action == PolicyAction.REDACT

    # __repr__ must NOT contain original_text
    assert "4532-0150-1234-5671" not in repr(scan_res)

    # safe_summary must NOT contain original_text
    summary = scan_res.safe_summary()
    summary_str = json.dumps(summary)
    assert "4532-0150-1234-5671" not in summary_str
    assert summary["overall_action"] == "REDACT"
    assert summary["integrity_hash"] == scan_res.integrity_hash

    # Serialization roundtrip
    serialized = scan_res.model_dump_json()
    reloaded = ScanResult.model_validate_json(serialized)
    assert reloaded.scan_id == scan_res.scan_id
    assert len(reloaded.detections) == 1
    assert reloaded.detections[0].entity_type == "PAYMENT_CARD"
    assert reloaded.integrity_hash == scan_res.integrity_hash
