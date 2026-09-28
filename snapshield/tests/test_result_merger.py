"""Comprehensive tests for result_merger module.

Validates:
1. Merging across:
   - Structured detection (Aadhaar, PAN, Phone, Email, Bank, IFSC, Card, IP)
   - Secret detection (API keys, passwords, JWT, AWS, Bearer, DB URIs)
   - Local AI contextual detection (PERSON, ADDRESS, MEDICAL_INFORMATION, etc.)
2. Exact duplicate removal:
   - Identical span and entity type merged with evidence combination
3. Overlapping span resolution:
   - Same entity type: merged into unified span with boosted confidence
4. Preserving different entity types when appropriate:
   - "Regex detects Phone. AI detects Person. Both must remain."
5. Preserving detector source and confidence:
   - Multi-detector corroboration preserved in detector string and explanation
   - Noisy-OR evidence accumulation for confidence
6. Positional sorting:
   - Returned list sorted strictly by start position
7. scan_text(text, ai_enabled=True / False) pipeline integration
"""

from __future__ import annotations

import pytest

from snapshield.core.models import Detection, SeverityLevel
from snapshield.core.result_merger import (
    ResultMerger,
    merge_detections,
    scan_text,
)


# =====================================================================
# 1. Exact Duplicate Removal & Evidence Merging Tests
# =====================================================================

def test_remove_exact_duplicates_and_merge_evidence():
    """Verify exact duplicates (same span and same category) are merged with accumulated evidence."""
    d1 = Detection(
        entity_type="Email",
        start=10,
        end=30,
        confidence=0.95,
        detector="regex_email",
        valid=True,
        severity=SeverityLevel.MEDIUM,
        explanation="Standard email address regex match.",
        raw_value="alice@example.com",
    )
    d2 = Detection(
        entity_type="EMAIL",
        start=10,
        end=30,
        confidence=0.85,
        detector="local_ai_dev_fallback",
        valid=True,
        severity=SeverityLevel.HIGH,
        explanation="Contextual email identification.",
        raw_value="alice@example.com",
    )

    merged = merge_detections([d1, d2])

    assert len(merged) == 1
    m = merged[0]
    # Span preserved
    assert m.start == 10
    assert m.end == 30
    # Both detector sources preserved
    assert "regex_email" in m.detector
    assert "local_ai_dev_fallback" in m.detector
    # Confidence combined (Noisy-OR: 1 - (1-0.95)*(1-0.85) = 0.9925)
    assert m.confidence > 0.95
    assert m.confidence == pytest.approx(0.9925, abs=0.001)
    # Severity upgraded to highest
    assert m.severity == SeverityLevel.HIGH
    # Explanation notes multi-detector corroboration
    assert "Corroborated by multiple detectors" in m.explanation


def test_remove_multiple_identical_duplicates():
    """Verify three identical detections collapse into one single corroborated detection."""
    d1 = Detection(
        entity_type="API_KEY",
        start=0,
        end=20,
        confidence=0.90,
        detector="detector_a",
        severity=SeverityLevel.HIGH,
        explanation="Detector A key match.",
    )
    d2 = Detection(
        entity_type="API_KEY",
        start=0,
        end=20,
        confidence=0.90,
        detector="detector_b",
        severity=SeverityLevel.CRITICAL,
        explanation="Detector B key match.",
    )
    d3 = Detection(
        entity_type="API_KEY",
        start=0,
        end=20,
        confidence=0.90,
        detector="detector_c",
        severity=SeverityLevel.HIGH,
        explanation="Detector C key match.",
    )

    merged = merge_detections([d1, d2, d3])
    assert len(merged) == 1
    assert "detector_a" in merged[0].detector
    assert "detector_b" in merged[0].detector
    assert "detector_c" in merged[0].detector
    assert merged[0].severity == SeverityLevel.CRITICAL
    assert merged[0].confidence > 0.95


# =====================================================================
# 2. Overlapping Spans of Same Entity Type
# =====================================================================

def test_resolve_overlapping_spans_same_entity_type():
    """Verify overlapping detections of the same entity type merge into a unified span."""
    # Suppose detector 1 found tokens 10:25, and detector 2 found tokens 15:35
    d1 = Detection(
        entity_type="ADDRESS",
        start=10,
        end=25,
        confidence=0.80,
        detector="ai_detector",
        severity=SeverityLevel.MEDIUM,
        explanation="AI address detection.",
    )
    d2 = Detection(
        entity_type="Address",
        start=15,
        end=35,
        confidence=0.75,
        detector="ner_secondary",
        severity=SeverityLevel.MEDIUM,
        explanation="Secondary address match.",
    )

    merged = merge_detections([d1, d2])
    assert len(merged) == 1
    m = merged[0]
    assert m.start == 10
    assert m.end == 35
    assert "ai_detector" in m.detector
    assert "ner_secondary" in m.detector
    assert m.confidence >= 0.80


# =====================================================================
# 3. Preserving Different Entity Types (Crucial Rule)
# =====================================================================

def test_preserve_different_entity_types_phone_and_person():
    """Verify the explicit specification requirement:

    Example:
    Regex detects Phone.
    AI detects Person.
    Both must remain.
    """
    phone_det = Detection(
        entity_type="Phone",
        start=15,
        end=25,
        confidence=0.95,
        detector="regex_indian_phone",
        valid=True,
        severity=SeverityLevel.HIGH,
        explanation="Standard phone number match.",
        raw_value="9876543210",
    )
    person_det = Detection(
        entity_type="PERSON",
        start=0,
        end=14,
        confidence=0.88,
        detector="local_ai_onnx",
        valid=True,
        severity=SeverityLevel.MEDIUM,
        explanation="Local AI contextual person detection.",
        raw_value="Dr. Alice Gray",
    )

    merged = merge_detections([phone_det, person_det])

    assert len(merged) == 2
    types = [d.entity_type for d in merged]
    assert "PERSON" in types
    assert "Phone" in types

    # Sorted by start position (Person at 0, Phone at 15)
    assert merged[0].entity_type == "PERSON"
    assert merged[1].entity_type == "Phone"


def test_preserve_different_entity_types_on_overlapping_spans():
    """Verify that even when spans overlap, DIFFERENT entity types both remain."""
    # E.g. AI tagged a broad span [0:30] as Person, and Regex tagged [15:25] as Phone
    broad_person = Detection(
        entity_type="PERSON",
        start=0,
        end=30,
        confidence=0.82,
        detector="local_ai_dev_fallback",
        severity=SeverityLevel.MEDIUM,
        explanation="Broad person context.",
        raw_value="Dr. Alice Gray 9876543210",
    )
    nested_phone = Detection(
        entity_type="Phone",
        start=15,
        end=25,
        confidence=0.98,
        detector="regex_indian_phone",
        severity=SeverityLevel.HIGH,
        explanation="Nested phone number.",
        raw_value="9876543210",
    )

    merged = merge_detections([broad_person, nested_phone])

    # Both must remain because they represent different entity types!
    assert len(merged) == 2
    types = {d.entity_type for d in merged}
    assert "PERSON" in types
    assert "Phone" in types


def test_preserve_different_entity_types_database_and_ip():
    """Verify database connection string and internal IP address are both preserved."""
    db_det = Detection(
        entity_type="DATABASE_URI",
        start=0,
        end=55,
        confidence=0.99,
        detector="regex_database_uri",
        severity=SeverityLevel.CRITICAL,
        explanation="Database connection URI match.",
    )
    ip_det = Detection(
        entity_type="IP Address",
        start=25,
        end=35,
        confidence=0.95,
        detector="regex_ipv4",
        severity=SeverityLevel.LOW,
        explanation="IPv4 internal address.",
    )

    merged = merge_detections([db_det, ip_det])

    assert len(merged) == 2
    types = {d.entity_type for d in merged}
    assert "DATABASE_URI" in types
    assert "IP Address" in types


# =====================================================================
# 4. Positional Sorting Verification
# =====================================================================

def test_sorting_by_position_strictly_enforced():
    """Verify that returned detections are strictly ordered by character start position."""
    d_late = Detection(
        entity_type="EMAIL",
        start=100,
        end=120,
        confidence=0.95,
        detector="det_1",
        explanation="Late email.",
    )
    d_early = Detection(
        entity_type="PAN",
        start=5,
        end=15,
        confidence=0.99,
        detector="det_2",
        explanation="Early PAN.",
    )
    d_mid = Detection(
        entity_type="PHONE",
        start=40,
        end=50,
        confidence=0.90,
        detector="det_3",
        explanation="Mid phone.",
    )

    # Pass in random/reverse order
    merged = merge_detections([d_late, d_early, d_mid])

    assert len(merged) == 3
    assert merged[0].start == 5
    assert merged[1].start == 40
    assert merged[2].start == 100


# =====================================================================
# 5. Full Pipeline scan_text() Integration Tests
# =====================================================================

def test_scan_text_mixed_synthetic_document_ai_enabled():
    """Verify scan_text executes structured, secret, and local AI detection together."""
    synthetic_doc = (
        "CONFIDENTIAL: Internal patient case notes.\n"
        "Patient ID: MRN-55442-X\n"
        "Attending physician: Dr. Benjamin Miller\n"
        "Contact email: dr.miller@synthetic-hospital.org\n"
        "Direct emergency phone: +91 9876543210\n"
        "Internal Database: postgres://admin:SuperSecretPass999@10.0.1.5:5432/patients\n"
        "Clinical diagnosis: Diagnosed with acute symptoms and started chemotherapy.\n"
        "API Authorization Key: sk-proj-syntheticKeyAlphaBeta1234567890abcdef\n"
    )

    detections = scan_text(synthetic_doc, ai_enabled=True)

    types = {d.entity_type for d in detections}

    # 1. Structured detections
    assert "Email" in types or "EMAIL" in types
    assert "Phone" in types or "PHONE" in types
    assert "IP Address" in types or "DATABASE_URI" in types

    # 2. Secret detections
    assert "DATABASE_URI" in types
    assert "API_KEY" in types

    # 3. Local AI contextual detections
    assert "PATIENT_IDENTIFIER" in types
    assert "PERSON" in types
    assert "MEDICAL_INFORMATION" in types
    assert "CONFIDENTIAL_BUSINESS_INFORMATION" in types

    # Verify positional sorting
    for i in range(len(detections) - 1):
        assert detections[i].start <= detections[i + 1].start

    # Verify no raw secrets leaked in repr or string representation
    for d in detections:
        assert "SuperSecretPass999" not in repr(d)
        assert "syntheticKeyAlphaBeta" not in repr(d)


def test_scan_text_ai_disabled():
    """Verify scan_text skips local AI detection when ai_enabled=False."""
    text = (
        "Dr. Alice Smith was diagnosed with acute symptoms.\n"
        "Contact email: alice@example.com\n"
        "API Key: sk-proj-syntheticTestingKey1234567890abcdefABCDEF\n"
    )

    detections = scan_text(text, ai_enabled=False)
    types = {d.entity_type for d in detections}

    # Structured and secret should be present
    assert "Email" in types or "EMAIL" in types
    assert "API_KEY" in types

    # AI contextual categories must NOT be present
    assert "MEDICAL_INFORMATION" not in types
    assert "PERSON" not in types


def test_result_merger_class_wrapper():
    """Verify ResultMerger object-oriented wrapper operates correctly."""
    merger = ResultMerger()
    text = "Send alert to +91 9876543210 and admin@network.internal"
    results = merger.scan_text(text, ai_enabled=False)

    assert len(results) == 2
    types = {r.entity_type for r in results}
    assert "Phone" in types
    assert "Email" in types


def test_scan_text_empty_and_whitespace():
    """Verify scan_text returns empty list for blank or invalid inputs."""
    assert scan_text("") == []
    assert scan_text("   \n\t  ") == []
    assert scan_text(None) == []  # type: ignore
    assert scan_text(999) == []  # type: ignore
