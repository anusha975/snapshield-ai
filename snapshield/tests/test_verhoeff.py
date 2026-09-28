"""Comprehensive unit tests for Verhoeff checksum algorithm and Aadhaar validation.

Enforces:
- Rule 7: Zero usage of real personal identifiers; strictly synthetic test data.
- Removal of spaces and hyphens.
- Exact 12-digit length requirement.
- Verhoeff checksum validation.
- Deterministic rejection of malformed values.
"""

from __future__ import annotations

import pytest

from snapshield.core.verhoeff import (
    clean_numeric_string,
    compute_verhoeff_check_digit,
    generate_synthetic_aadhaar,
    is_valid_aadhaar,
    is_valid_verhoeff,
)


# ==========================================
# 1. Verhoeff Checksum Algorithm Tests
# ==========================================

def test_verhoeff_valid_synthetic_numbers():
    """Verify known mathematically valid Verhoeff numbers pass validation."""
    # Test number: 23456789012 + check digit 4 -> 234567890124
    assert is_valid_verhoeff("234567890124") is True
    # With spaces and hyphens
    assert is_valid_verhoeff("2345 6789 0124") is True
    assert is_valid_verhoeff("2345-6789-0124") is True
    assert is_valid_verhoeff("  2345--6789  0124  ") is True


def test_verhoeff_tampered_check_digit_fails():
    """Verify that altering the check digit causes checksum failure."""
    valid_number = "234567890124"
    for digit in "012356789":  # All digits except the correct check digit '4'
        tampered = valid_number[:-1] + digit
        assert is_valid_verhoeff(tampered) is False


def test_verhoeff_single_digit_transcription_error():
    """Verify Verhoeff mathematically detects single-digit substitution errors."""
    valid_number = "234567890124"
    # Mutate digit at position 3 ('5' -> '9')
    mutated = valid_number[:3] + "9" + valid_number[4:]
    assert is_valid_verhoeff(mutated) is False


def test_verhoeff_adjacent_transposition_error():
    """Verify Verhoeff mathematically detects adjacent transposition errors (e.g. 56 -> 65)."""
    valid_number = "234567890124"
    # Swap '5' and '6' at index 3 and 4 -> 234657890124
    transposed = "234657890124"
    assert is_valid_verhoeff(transposed) is False


def test_verhoeff_malformed_inputs():
    """Verify Verhoeff rejects malformed, non-numeric, or empty inputs."""
    assert is_valid_verhoeff("") is False
    assert is_valid_verhoeff("   ") is False
    assert is_valid_verhoeff("---") is False
    assert is_valid_verhoeff("23456789012A") is False
    assert is_valid_verhoeff("ABCDEFGHIJKL") is False
    assert is_valid_verhoeff(None) is False  # type: ignore


# ==========================================
# 2. Aadhaar Validation Tests (Synthetic Only)
# ==========================================

def test_aadhaar_valid_synthetic_samples():
    """Verify valid synthetic Aadhaar numbers across diverse valid starting digits (2-9)."""
    # Generate deterministic synthetic test cases
    seeds = [
        "23456789012",
        "34567890123",
        "45678901234",
        "56789012345",
        "67890123456",
        "78901234567",
        "89012345678",
        "90123456789",
    ]

    for seed in seeds:
        synthetic_aadhaar = generate_synthetic_aadhaar(seed)
        assert len(synthetic_aadhaar) == 12
        assert is_valid_aadhaar(synthetic_aadhaar) is True

        # Formatted with spaces
        spaced = f"{synthetic_aadhaar[:4]} {synthetic_aadhaar[4:8]} {synthetic_aadhaar[8:]}"
        assert is_valid_aadhaar(spaced) is True

        # Formatted with hyphens
        hyphenated = f"{synthetic_aadhaar[:4]}-{synthetic_aadhaar[4:8]}-{synthetic_aadhaar[8:]}"
        assert is_valid_aadhaar(hyphenated) is True


def test_aadhaar_rejects_invalid_lengths():
    """Verify Aadhaar requires exactly 12 digits after cleaning."""
    # 11 digits
    assert is_valid_aadhaar("23456789012") is False
    # 13 digits
    assert is_valid_aadhaar("2345678901245") is False
    # 4 digits
    assert is_valid_aadhaar("2345") is False


def test_aadhaar_rejects_leading_zero_or_one():
    """Verify UIDAI rule: Aadhaar numbers must not start with 0 or 1."""
    # Seed starting with '0'
    check_zero = compute_verhoeff_check_digit("01234567890")
    candidate_zero = f"01234567890{check_zero}"
    # Verhoeff passes, but Aadhaar policy must reject leading 0
    assert is_valid_verhoeff(candidate_zero) is True
    assert is_valid_aadhaar(candidate_zero) is False

    # Seed starting with '1'
    check_one = compute_verhoeff_check_digit("12345678901")
    candidate_one = f"12345678901{check_one}"
    # Verhoeff passes, but Aadhaar policy must reject leading 1
    assert is_valid_verhoeff(candidate_one) is True
    assert is_valid_aadhaar(candidate_one) is False


def test_aadhaar_rejects_all_identical_digits():
    """Verify identical repeated digits (e.g. 2222 2222 2222) are rejected."""
    assert is_valid_aadhaar("0000 0000 0000") is False
    assert is_valid_aadhaar("1111-1111-1111") is False
    assert is_valid_aadhaar("2222 2222 2222") is False
    assert is_valid_aadhaar("9999 9999 9999") is False


def test_aadhaar_rejects_alphanumeric_and_special_chars():
    """Verify rejection of strings containing letters or symbols."""
    assert is_valid_aadhaar("2345-6789-012A") is False
    assert is_valid_aadhaar("2345#6789#0124") is False
    assert is_valid_aadhaar("ABCD-EFGH-IJKL") is False


def test_aadhaar_rejects_empty_and_none():
    """Verify rejection of empty or None inputs without throwing exceptions."""
    assert is_valid_aadhaar("") is False
    assert is_valid_aadhaar("   ") is False
    assert is_valid_aadhaar(None) is False  # type: ignore


# ==========================================
# 3. Synthetic Generator Tests
# ==========================================

def test_synthetic_aadhaar_generator():
    """Verify synthetic generator generates valid Aadhaar numbers and validates input seed."""
    val = generate_synthetic_aadhaar("98765432109")
    assert len(val) == 12
    assert is_valid_aadhaar(val) is True

    # Invalid seed: wrong length
    with pytest.raises(ValueError) as exc:
        generate_synthetic_aadhaar("12345")
    assert "Seed must be exactly 11 numeric digits" in str(exc.value)

    # Invalid seed: starts with 0 or 1
    with pytest.raises(ValueError):
        generate_synthetic_aadhaar("01234567890")

    with pytest.raises(ValueError):
        generate_synthetic_aadhaar("12345678901")
