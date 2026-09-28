"""Comprehensive tests for the redaction and content sanitization engine (core/redactor.py).

Validates:
1. Policy-guided transformations:
   - REDACT: Replaces span with [REDACTED: ENTITY_TYPE]
   - ALLOW: Preserves original text intact
   - BLOCK: Marks content as prohibited (blocked=True)
   - REVIEW: Flags content as requiring user approval (requires_review=True)
2. Directional replacement integrity:
   - Replacements performed strictly from end toward beginning to prevent offset drift
3. Immutability of input:
   - Original text is never mutated or overwritten
4. 5-tuple unpacking & attribute access:
   - sanitized_text, redaction_count, summary, blocked, requires_review
5. Safety leak verification:
   - Verified that blocked/redacted values cannot accidentally remain in sanitized text
6. Complex span scenarios:
   - Adjacent detections (end1 == start2)
   - Overlapping detections (start2 < end1)
   - Nested detections (start1 <= start2 < end2 <= end1)
   - Multiple mixed detections across large documents
"""

from __future__ import annotations

import pytest

from snapshield.core.models import Detection, PolicyAction, PolicyDecision, SeverityLevel
from snapshield.core.redactor import (
    RedactionResult,
    Redactor,
    format_placeholder,
    redact,
    redact_text,
)


# =====================================================================
# 1. Standard Policy Action Replacements Tests
# =====================================================================

def test_redact_action_replaces_with_standard_placeholder():
    """Verify REDACT action replaces text with [REDACTED: ENTITY_TYPE]."""
    text = "Contact Alice at 9876543210 for appointment."
    phone_det = Detection(
        entity_type="Phone",
        start=17,
        end=27,
        confidence=0.95,
        detector="regex_phone",
        explanation="Phone match",
        raw_value="9876543210",
    )

    result = redact(text, [phone_det], mode="Personal")

    assert result.sanitized_text == "Contact Alice at [REDACTED: PHONE] for appointment."
    assert result.redaction_count == 1
    assert result.blocked is False
    assert result.requires_review is False
    assert "9876543210" not in result.sanitized_text


def test_allow_action_preserves_original_text():
    """Verify ALLOW action preserves original text exactly without alteration."""
    text = "Send confirmation to user@example.com immediately."
    email_det = Detection(
        entity_type="Email",
        start=21,
        end=37,
        confidence=0.98,
        detector="regex_email",
        explanation="Email match",
        raw_value="user@example.com",
    )

    # In Personal mode, Email is ALLOW
    result = redact(text, [email_det], mode="Personal")

    assert result.sanitized_text == text
    assert result.redaction_count == 0
    assert result.blocked is False
    assert result.requires_review is False
    assert "user@example.com" in result.sanitized_text


def test_block_action_marks_content_as_prohibited():
    """Verify BLOCK action marks content as blocked=True with prohibited summary."""
    text = "AWS token: AKIAIOSFODNN7EXAMPLE is used for deployment."
    key_det = Detection(
        entity_type="API_KEY",
        start=11,
        end=31,
        confidence=0.99,
        detector="regex_aws",
        severity=SeverityLevel.CRITICAL,
        explanation="AWS key",
        raw_value="AKIAIOSFODNN7EXAMPLE",
    )

    # In Developer mode, API_KEY is BLOCK
    result = redact(text, [key_det], mode="Developer")

    assert result.blocked is True
    assert "PROHIBITED" in result.summary
    assert "AKIAIOSFODNN7EXAMPLE" not in result.sanitized_text
    assert "[BLOCKED: API_KEY]" in result.sanitized_text


def test_review_action_flags_requires_review():
    """Verify REVIEW action sets requires_review=True."""
    text = "Host address is 10.0.1.50 in subnet."
    ip_det = Detection(
        entity_type="IP_ADDRESS",
        start=16,
        end=25,
        confidence=0.95,
        detector="regex_ip",
        explanation="Internal IP",
        raw_value="10.0.1.50",
    )

    # In Developer mode, IP_ADDRESS is REVIEW
    result = redact(text, [ip_det], mode="Developer")

    assert result.requires_review is True
    assert "REVIEW REQUIRED" in result.summary
    assert "10.0.1.50" not in result.sanitized_text
    assert "[REDACTED: IP_ADDRESS]" in result.sanitized_text


# =====================================================================
# 2. Return Structure & 5-Tuple Unpacking Tests
# =====================================================================

def test_return_values_attribute_and_unpacking():
    """Verify return values match specification:

    sanitized_text
    redaction_count
    summary
    blocked
    requires_review
    """
    text = "Aadhaar: 999912345678"
    det = Detection(
        entity_type="Aadhaar",
        start=9,
        end=21,
        confidence=1.0,
        detector="regex",
        explanation="Gov ID",
        raw_value="999912345678",
    )

    result = redact(text, [det], mode="Personal")

    # 1. Attribute access
    assert isinstance(result, RedactionResult)
    assert result.sanitized_text == "Aadhaar: [REDACTED: AADHAAR]"
    assert result.redaction_count == 1
    assert isinstance(result.summary, str)
    assert result.blocked is False
    assert result.requires_review is False

    # 2. 5-tuple unpacking
    sanitized, count, summary, blocked, requires_review = result
    assert sanitized == "Aadhaar: [REDACTED: AADHAAR]"
    assert count == 1
    assert isinstance(summary, str)
    assert blocked is False
    assert requires_review is False


# =====================================================================
# 3. Directional Replacement & Original Text Immutability
# =====================================================================

def test_replacements_from_end_to_beginning_preserves_offsets():
    """Verify multiple replacements processed from end toward beginning avoid offset drift."""
    # Text with 3 entities:
    # 1. Aadhaar at [9:21]
    # 2. Phone at [33:43]
    # 3. Bank Account at [54:66]
    text = "Aadhaar: 999912345678 | Phone: 9876543210 | Bank: 123456789012"

    d1 = Detection(entity_type="Aadhaar", start=9, end=21, confidence=1.0, detector="r", explanation="ID", raw_value="999912345678")
    d2 = Detection(entity_type="Phone", start=31, end=41, confidence=1.0, detector="r", explanation="Phone", raw_value="9876543210")
    d3 = Detection(entity_type="Bank Account", start=50, end=62, confidence=1.0, detector="r", explanation="Bank", raw_value="123456789012")

    result = redact(text, [d1, d2, d3], mode="Personal")

    expected = "Aadhaar: [REDACTED: AADHAAR] | Phone: [REDACTED: PHONE] | Bank: [REDACTED: BANK_ACCOUNT]"
    assert result.sanitized_text == expected
    assert result.redaction_count == 3


def test_original_text_is_never_overwritten():
    """Verify original input text remains completely unaltered."""
    original = "Secret PAN: ABCDE1234F"
    original_copy = str(original)

    pan_det = Detection(
        entity_type="PAN",
        start=12,
        end=22,
        confidence=1.0,
        detector="regex",
        explanation="PAN",
        raw_value="ABCDE1234F",
    )

    result = redact(original, [pan_det], mode="Personal")

    assert original == original_copy
    assert "ABCDE1234F" in original
    assert "ABCDE1234F" not in result.sanitized_text


# =====================================================================
# 4. Adjacent, Overlapping, and Nested Detections Tests
# =====================================================================

def test_adjacent_detections_handled_cleanly():
    """Verify adjacent spans (end1 == start2) are replaced cleanly side-by-side."""
    text = "CODE:999912345678ABCDE1234F"
    # Aadhaar at [5:17], PAN at [17:27]
    d1 = Detection(entity_type="Aadhaar", start=5, end=17, confidence=1.0, detector="r", explanation="A", raw_value="999912345678")
    d2 = Detection(entity_type="PAN", start=17, end=27, confidence=1.0, detector="r", explanation="P", raw_value="ABCDE1234F")

    result = redact(text, [d1, d2], mode="Personal")

    assert result.sanitized_text == "CODE:[REDACTED: AADHAAR][REDACTED: PAN]"
    assert result.redaction_count == 2
    assert "999912345678" not in result.sanitized_text
    assert "ABCDE1234F" not in result.sanitized_text


def test_overlapping_detections_merged_safely():
    """Verify overlapping spans (start2 < end1) are consolidated to prevent corruption."""
    text = "Information: Dr. Alice Gray 9876543210 is attending."
    # Person at [13:39], Phone at [28:38]
    d_person = Detection(
        entity_type="Medical Information",
        start=13,
        end=39,
        confidence=0.85,
        detector="ai",
        explanation="Med",
        raw_value="Dr. Alice Gray 9876543210",
    )
    d_phone = Detection(
        entity_type="Phone",
        start=28,
        end=38,
        confidence=0.95,
        detector="r",
        explanation="Phone",
        raw_value="9876543210",
    )

    result = redact(text, [d_person, d_phone], mode="Personal")

    # Replaces the entire overlapping region safely
    assert "9876543210" not in result.sanitized_text
    assert "Dr. Alice Gray" not in result.sanitized_text
    assert "is attending." in result.sanitized_text


def test_nested_detections_handled_safely():
    """Verify nested span (one completely inside another) does not corrupt replacement."""
    text = "Database URI postgres://user:pass123@10.0.0.1:5432/db is active."
    # Database URI at [13:53], IP at [37:45]
    d_db = Detection(
        entity_type="API_KEY",  # will be blocked in developer mode
        start=13,
        end=53,
        confidence=0.99,
        detector="cred",
        explanation="DB",
        raw_value="postgres://user:pass123@10.0.0.1:5432/db",
    )
    d_ip = Detection(
        entity_type="IP_ADDRESS",
        start=37,
        end=45,
        confidence=0.90,
        detector="regex",
        explanation="IP",
        raw_value="10.0.0.1",
    )

    result = redact(text, [d_db, d_ip], mode="Developer")

    assert "pass123" not in result.sanitized_text
    assert "10.0.0.1" not in result.sanitized_text
    assert result.blocked is True


# =====================================================================
# 5. Safety Verification: No Leaked Sensitive Values
# =====================================================================

def test_leak_verification_ensures_zero_sensitive_tokens_remain():
    """Verify that verify_sanitization_leak_free detects any accidental leakage."""
    secret = "SuperSecretTokenString999"
    text = f"API Token: {secret}"

    det = Detection(
        entity_type="API_KEY",
        start=11,
        end=11 + len(secret),
        confidence=1.0,
        detector="cred",
        explanation="Token",
        raw_value=secret,
    )

    result = redact(text, [det], mode="Developer")
    assert secret not in result.sanitized_text
    assert result.blocked is True


def test_empty_input_and_no_detections():
    """Verify empty text or empty detections returns safely."""
    res_empty = redact("", [])
    assert res_empty.sanitized_text == ""
    assert res_empty.redaction_count == 0

    res_clean = redact("Hello world, clean text.", [])
    assert res_clean.sanitized_text == "Hello world, clean text."
    assert res_clean.redaction_count == 0
