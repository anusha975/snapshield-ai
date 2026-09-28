"""Comprehensive unit tests for deterministic sensitive-data detection.

Validates:
1. Aadhaar detection (Regex candidate -> Normalization -> Verhoeff validation)
2. PAN detection (structural validation, 4th character entity category)
3. Indian phone numbers (+91, 0, 10 digits starting with 6-9)
4. Email (RFC-compliant)
5. Bank account (requires contextual confirmation; avoids blind numeric detection)
6. IFSC (11 chars, 5th char '0')
7. Card-like numbers (Luhn Mod-10 checksum validation)
8. IP addresses (IPv4 valid octets 0-255)
9. False positives and malformed input rejection
10. Detection attributes: type, start, end, confidence, detector, valid, severity, explanation
"""

from __future__ import annotations

import pytest

from snapshield.core.detectors import (
    detect_structured,
    is_valid_ifsc,
    is_valid_ipv4,
    is_valid_pan,
    luhn_checksum_validate,
)
from snapshield.core.models import Detection, SeverityLevel
from snapshield.core.verhoeff import generate_synthetic_aadhaar


# ==========================================
# 1. Aadhaar Detection Tests
# ==========================================

def test_detect_valid_synthetic_aadhaar():
    """Verify detection of valid synthetic Aadhaar number with Verhoeff checksum."""
    aadhaar_num = generate_synthetic_aadhaar("23456789012")  # '234567890124'
    text = f"Customer national ID is {aadhaar_num} for KYC verification."

    findings = detect_structured(text)
    aadhaar_findings = [f for f in findings if f.entity_type == "Aadhaar"]

    assert len(aadhaar_findings) == 1
    f = aadhaar_findings[0]
    assert f.type == "Aadhaar"
    assert f.confidence == 1.0
    assert f.detector == "regex+verhoeff"
    assert f.valid is True
    assert f.severity == SeverityLevel.HIGH
    assert f.start == text.find(aadhaar_num)
    assert f.end == f.start + len(aadhaar_num)
    assert "Verhoeff checksum" in f.explanation


def test_aadhaar_false_positive_rejection():
    """Verify that 12-digit numbers with invalid Verhoeff checksum are NOT classified as Aadhaar."""
    # 234567890120 (check digit is 4, so 0 is invalid)
    invalid_aadhaar = "234567890120"
    text = f"Reference transaction number: {invalid_aadhaar}."

    findings = detect_structured(text)
    aadhaar_findings = [f for f in findings if f.entity_type == "Aadhaar"]
    assert len(aadhaar_findings) == 0


def test_aadhaar_formatted_with_spaces_or_hyphens():
    """Verify formatted synthetic Aadhaar (XXXX XXXX XXXX or XXXX-XXXX-XXXX) is detected."""
    raw = generate_synthetic_aadhaar("34567890123")
    spaced = f"{raw[:4]} {raw[4:8]} {raw[8:]}"
    text = f"UIDAI copy: {spaced}"

    findings = detect_structured(text)
    assert len(findings) == 1
    assert findings[0].entity_type == "Aadhaar"
    assert findings[0].valid is True


# ==========================================
# 2. PAN (Permanent Account Number) Tests
# ==========================================

def test_detect_valid_pan():
    """Verify PAN detection with valid structure (5 letters, 4 digits, 1 letter)."""
    text = "Taxpayer PAN details: ABCPE1234F submitted."
    findings = detect_structured(text)

    pan_findings = [f for f in findings if f.entity_type == "PAN"]
    assert len(pan_findings) == 1
    f = pan_findings[0]
    assert f.type == "PAN"
    assert f.confidence == 0.95
    assert f.detector == "regex_structural"
    assert f.valid is True
    assert f.severity == SeverityLevel.HIGH


def test_pan_false_positive_rejection():
    """Verify rejection of malformed or invalid PAN strings."""
    # 4th character must be in [ABCFGHLJPT]; 'Z' is invalid
    invalid_status_pan = "ABCDZ1234F"
    # Too many digits
    wrong_digits_pan = "ABCDE12345F"

    text = f"Codes: {invalid_status_pan} and {wrong_digits_pan}."
    findings = detect_structured(text)
    pan_findings = [f for f in findings if f.entity_type == "PAN"]
    assert len(pan_findings) == 0


# ==========================================
# 3. Indian Phone Numbers Tests
# ==========================================

def test_detect_valid_indian_phone():
    """Verify Indian phone detection with +91, 0, or direct 10-digit formats."""
    text = "Call helpline at +91 9876543210 or alternate 08765432109 for support."
    findings = detect_structured(text)

    phone_findings = [f for f in findings if f.entity_type == "Phone"]
    assert len(phone_findings) == 2
    for p in phone_findings:
        assert p.type == "Phone"
        assert p.confidence == 0.90
        assert p.detector == "regex_prefix"
        assert p.valid is True


def test_phone_false_positive_rejection():
    """Verify rejection of numbers starting with 1-5 or repetitive dummy sequences."""
    text = "Order tracking ID: 5551234567 and dummy: 9999999999."
    findings = detect_structured(text)

    phone_findings = [f for f in findings if f.entity_type == "Phone"]
    assert len(phone_findings) == 0


# ==========================================
# 4. Email Address Tests
# ==========================================

def test_detect_valid_email():
    """Verify RFC email detection."""
    text = "Contact security officer at security-ops@synthetic-bank.org for disclosures."
    findings = detect_structured(text)

    email_findings = [f for f in findings if f.entity_type == "Email"]
    assert len(email_findings) == 1
    f = email_findings[0]
    assert f.type == "Email"
    assert f.confidence == 0.95
    assert f.detector == "rfc5322_regex"
    assert f.severity == SeverityLevel.LOW


# ==========================================
# 5. Bank Account Tests (Contextual Verification)
# ==========================================

def test_detect_bank_account_with_financial_context():
    """Verify bank account is identified when surrounded by financial context keywords."""
    text = "Please credit the savings account: 98765432101234 before end of day."
    findings = detect_structured(text)

    bank_findings = [f for f in findings if f.entity_type == "Bank Account"]
    assert len(bank_findings) == 1
    f = bank_findings[0]
    assert f.type == "Bank Account"
    assert f.confidence == 0.90
    assert f.detector == "contextual_regex"
    assert f.severity == SeverityLevel.HIGH


def test_bank_account_false_positive_rejection_without_context():
    """Verify that raw numbers without financial context are NOT classified as bank accounts."""
    # A standalone 14-digit hardware part number or order ID
    text = "Hardware component serial number: 98765432101234 (rev B)."
    findings = detect_structured(text)

    bank_findings = [f for f in findings if f.entity_type == "Bank Account"]
    assert len(bank_findings) == 0


# ==========================================
# 6. IFSC (Indian Financial System Code) Tests
# ==========================================

def test_detect_valid_ifsc():
    """Verify IFSC code detection (4 letters, 0, 6 alphanumeric)."""
    text = "Branch IFSC code is SBIN0001234 for RTGS wire."
    findings = detect_structured(text)

    ifsc_findings = [f for f in findings if f.entity_type == "IFSC"]
    assert len(ifsc_findings) == 1
    f = ifsc_findings[0]
    assert f.type == "IFSC"
    assert f.confidence == 0.95
    assert f.detector == "regex_structural"


def test_ifsc_false_positive_rejection():
    """Verify IFSC rejection when 5th character is not '0'."""
    # 5th character is '1', invalid IFSC
    text = "Code SBIN1001234 is invalid."
    findings = detect_structured(text)
    ifsc_findings = [f for f in findings if f.entity_type == "IFSC"]
    assert len(ifsc_findings) == 0


# ==========================================
# 7. Card-Like Numbers (Luhn Validation) Tests
# ==========================================

def test_detect_valid_card_number():
    """Verify payment card detection with valid Luhn checksum."""
    card = "4532 0150 1234 5671"
    text = f"Payment processed on Visa card {card} successfully."

    findings = detect_structured(text)
    card_findings = [f for f in findings if f.entity_type == "Card"]
    assert len(card_findings) == 1

    f = card_findings[0]
    assert f.type == "Card"
    assert f.confidence == 1.0
    assert f.detector == "regex+luhn"
    assert f.valid is True
    assert f.severity == SeverityLevel.CRITICAL


def test_card_false_positive_rejection():
    """Verify that numbers failing Luhn are NOT detected as Card."""
    # Card with tampered check digit (5671 -> 5672)
    invalid_card = "4532 0150 1234 5672"
    text = f"Transaction code: {invalid_card}."

    findings = detect_structured(text)
    card_findings = [f for f in findings if f.entity_type == "Card"]
    assert len(card_findings) == 0


# ==========================================
# 8. IP Address Tests
# ==========================================

def test_detect_valid_ipv4():
    """Verify IPv4 address detection."""
    text = "Gateway router configured at 192.168.1.254 and DNS at 10.0.0.1."
    findings = detect_structured(text)

    ip_findings = [f for f in findings if f.entity_type == "IP Address"]
    assert len(ip_findings) == 2
    for ip in ip_findings:
        assert ip.type == "IP Address"
        assert ip.confidence == 0.95
        assert ip.detector == "regex_ipv4"


def test_ipv4_false_positive_rejection():
    """Verify rejection of invalid octets (> 255 or leading zeros)."""
    text = "Invalid IPs: 999.1.2.3 and 10.01.2.3 and 1.2.3."
    findings = detect_structured(text)
    ip_findings = [f for f in findings if f.entity_type == "IP Address"]
    assert len(ip_findings) == 0


# ==========================================
# 9. Mixed Complex Document Test
# ==========================================

def test_detect_structured_mixed_document():
    """Verify simultaneous multi-entity detection without span collisions."""
    aadhaar = generate_synthetic_aadhaar("45678901234")
    sample_text = (
        f"Beneficiary KYC: Jane Doe, PAN: ABCPE1234F, Aadhaar: {aadhaar}.\n"
        f"Bank account: 123456789012, IFSC: HDFC0001234.\n"
        f"Card: 4532-0150-1234-5671, Email: jane@synthetic-corp.io, Phone: +91 9876543210.\n"
        f"Server IP: 10.14.0.1."
    )

    findings = detect_structured(sample_text)
    detected_types = {f.type for f in findings}

    assert "Aadhaar" in detected_types
    assert "PAN" in detected_types
    assert "Bank Account" in detected_types
    assert "IFSC" in detected_types
    assert "Card" in detected_types
    assert "Email" in detected_types
    assert "Phone" in detected_types
    assert "IP Address" in detected_types

    # Ensure all required attributes are present and typed
    for f in findings:
        assert isinstance(f, Detection)
        assert f.start >= 0
        assert f.end > f.start
        assert 0.0 <= f.confidence <= 1.0
        assert f.valid is True
        assert isinstance(f.severity, SeverityLevel)
        assert len(f.explanation) > 0
        assert f.get_masked_display_value() != ""
