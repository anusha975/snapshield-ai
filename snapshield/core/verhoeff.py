"""Verhoeff Checksum Algorithm and Aadhaar Number Validator.

Implements:
- Standard Verhoeff dihedral D_5 checksum validation.
- Check digit computation for synthetic data generation.
- Strict Aadhaar validation (12 digits, removal of formatting hyphens/spaces,
  leading digit check, Verhoeff checksum verification).
- Deterministic, UI-independent functions.
- Zero usage of real personal identifiers (synthetic generation supported).
"""

from __future__ import annotations

import re
from typing import Optional


# Multiplication table (d) based on dihedral group D_5
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table (p)
VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table (inv)
VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

_RE_CLEAN_NUMERIC = re.compile(r"[\s\-]+")


def clean_numeric_string(number: str) -> str:
    """Strips common formatting whitespace and hyphens from numeric strings."""
    if not isinstance(number, str):
        return ""
    # Remove spaces and hyphens using precompiled regex
    return _RE_CLEAN_NUMERIC.sub("", number.strip())


def is_valid_verhoeff(number: str) -> bool:
    """Validates whether a number string satisfies the Verhoeff checksum algorithm.

    Requirements:
    - Removes whitespace and hyphens.
    - Requires all characters to be digits.
    - Evaluates dihedral group permutation checksum.
    - Rejects empty, non-digit, or corrupt inputs.

    Args:
        number: The input string containing the digits and check digit.

    Returns:
        bool: True if checksum validates to 0, False otherwise.
    """
    cleaned = clean_numeric_string(number)

    if not cleaned or not cleaned.isdigit():
        return False

    c = 0
    # Process digits from right to left (index 0 corresponds to the check digit)
    for i, char in enumerate(reversed(cleaned)):
        digit = int(char)
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]

    return c == 0


def compute_verhoeff_check_digit(number: str) -> Optional[int]:
    """Computes the Verhoeff check digit for an input digit sequence.

    Useful for constructing deterministic synthetic test data without real PII.
    """
    cleaned = clean_numeric_string(number)
    if not cleaned or not cleaned.isdigit():
        return None

    c = 0
    # Process digits from right to left with position starting at 1
    for i, char in enumerate(reversed(cleaned), start=1):
        digit = int(char)
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]

    return VERHOEFF_INV[c]


def is_valid_aadhaar(number: str) -> bool:
    """Validates an Indian Aadhaar identification number.

    Requirements:
    - Removes spaces and hyphens (supports formats: 'XXXX-XXXX-XXXX' or 'XXXX XXXX XXXX' or 'XXXXXXXXXXXX').
    - Requires exactly 12 numeric digits.
    - Must not start with '0' or '1' (valid Aadhaar numbers begin with digits 2-9).
    - Must not consist of repeated identical digits (e.g., '000000000000' or '111111111111').
    - Validates against the Verhoeff checksum algorithm.
    - Rejects malformed values deterministically.

    Args:
        number: The candidate Aadhaar number string.

    Returns:
        bool: True if the candidate is a structurally and mathematically valid Aadhaar number.
    """
    cleaned = clean_numeric_string(number)

    # 1. Require exactly 12 digits
    if len(cleaned) != 12:
        return False

    # 2. Require strictly numeric digits
    if not cleaned.isdigit():
        return False

    # 3. UIDAI standard: First digit cannot be 0 or 1
    if cleaned[0] in ("0", "1"):
        return False

    # 4. Reject all identical digits (e.g. 222222222222)
    if len(set(cleaned)) == 1:
        return False

    # 5. Validate Verhoeff checksum
    return is_valid_verhoeff(cleaned)


def generate_synthetic_aadhaar(seed_11_digits: str = "23456789012") -> str:
    """Generates a mathematically valid synthetic 12-digit Aadhaar number for testing.

    Guarantees no real personal identifier is used while preserving algorithm integrity.
    """
    cleaned = clean_numeric_string(seed_11_digits)
    if len(cleaned) != 11 or not cleaned.isdigit() or cleaned[0] in ("0", "1"):
        raise ValueError("Seed must be exactly 11 numeric digits starting with 2-9.")

    check_digit = compute_verhoeff_check_digit(cleaned)
    if check_digit is None:
        raise ValueError("Failed to compute check digit for seed.")

    return f"{cleaned}{check_digit}"
