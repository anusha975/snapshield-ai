"""Deterministic sensitive-data detection engine for SnapShield AI.

Features:
- Deterministic regex + algorithmic checksum validation.
- Specific categories:
  1. Aadhaar (Regex candidate -> normalization -> Verhoeff validation)
  2. PAN (Permanent Account Number structural pattern)
  3. Indian phone numbers (+91, 10 digits starting with 6-9)
  4. Email (RFC-compliant addresses)
  5. Bank account (9-18 digits with contextual confirmation)
  6. IFSC (11 chars, 5th char '0')
  7. Card-like numbers (Luhn Mod-10 checksum validation)
  8. IP addresses (IPv4 valid octets 0-255)
- Does not classify arbitrary numbers as sensitive.
- Never logs raw matched values (enforces Rule 2).
- Returns strongly typed list[Detection].
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from snapshield.core.logger import logger
from snapshield.core.models import Detection, SeverityLevel
from snapshield.core.verhoeff import clean_numeric_string, is_valid_aadhaar


# ==========================================
# Checksum Algorithms & Validation Utilities
# ==========================================

def luhn_checksum_validate(number_str: str) -> bool:
    """Validates payment card numbers using the standard Luhn Mod-10 algorithm."""
    clean_digits = [int(c) for c in number_str if c.isdigit()]
    if len(clean_digits) < 13 or len(clean_digits) > 19:
        return False

    checksum = 0
    reverse_digits = clean_digits[::-1]

    for idx, digit in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit

    return checksum % 10 == 0


# Precompiled validation regexes
_RE_PAN_EXACT = re.compile(r"^[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]$")
_RE_IFSC_EXACT = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def is_valid_pan(pan_str: str) -> bool:
    """Validates Indian Permanent Account Number (PAN) structure:

    Format: [A-Z]{3}[ABCFGHLJPT][A-Z]{1}[0-9]{4}[A-Z]{1}
    4th character designates taxpayer category.
    """
    clean_pan = pan_str.strip().upper()
    if len(clean_pan) != 10:
        return False
    return bool(_RE_PAN_EXACT.match(clean_pan))


def is_valid_ifsc(ifsc_str: str) -> bool:
    """Validates Indian Financial System Code (IFSC):

    Format: 4 alphabetic chars + '0' + 6 alphanumeric chars.
    """
    clean_ifsc = ifsc_str.strip().upper()
    if len(clean_ifsc) != 11:
        return False
    return bool(_RE_IFSC_EXACT.match(clean_ifsc))


def is_valid_ipv4(ip_str: str) -> bool:
    """Validates an IPv4 address where each octet is strictly between 0 and 255."""
    parts = ip_str.strip().split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit() or not 0 <= int(part) <= 255:
            return False
        # Avoid leading zeroes (e.g. 01.02.03.04) unless single '0'
        if len(part) > 1 and part.startswith("0"):
            return False
    return True


# ==========================================
# Deterministic Patterns
# ==========================================

# 1. Aadhaar Candidate: 12 digits, optional spaces or hyphens
RE_AADHAAR_CANDIDATE = re.compile(r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b")

# 2. PAN Candidate: 5 letters, 4 digits, 1 letter
RE_PAN = re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b")

# 3. Indian Phone: Optional +91 / 91 / 0 followed by 10 digits starting with 6-9
RE_INDIAN_PHONE = re.compile(r"\b(?:\+91[-\s]?|91[-\s]?|0)?[6-9]\d{4}[-\s]?\d{5}\b")

# 4. Email: RFC-5322 compatible
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 5. Bank Account Context & Candidate: requires keyword within proximity
RE_BANK_CONTEXT = re.compile(r"(?i)\b(?:account|acc(?:\s*no)?|a/c|acct|bank|savings|current|beneficiary|deposit)\b")
RE_BANK_ACCOUNT_DIGITS = re.compile(r"\b\d{9,18}\b")

# 6. IFSC Candidate: 4 letters, 0, 6 alphanumeric
RE_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")

# 7. Card-like numbers: 13 to 19 digits with optional groups of 4
RE_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{1,7}\b")

# 8. IP Address (IPv4)
RE_IPV4 = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


# ==========================================
# Detection Engine
# ==========================================

def detect_structured(text: str) -> list[Detection]:
    """Scans input text and returns structured sensitive detections.

    Enforces:
    - Zero logging of raw sensitive matches.
    - Checksum and structural verification before classification.
    - Contextual confirmation for non-checksum numeric entities (e.g. bank accounts).
    - Accurate positional offsets [start:end].
    - Safe display helpers that mask raw values.

    Args:
        text: Input text string to inspect.

    Returns:
        list[Detection]: Discovered sensitive entities with type, span, confidence, and severity.
    """
    if not text or not isinstance(text, str):
        return []

    detections: list[Detection] = []
    occupied_spans: list[Tuple[int, int]] = []

    def _is_overlapping(start: int, end: int) -> bool:
        """Returns True if the candidate span overlaps with an already validated higher-priority finding."""
        for s, e in occupied_spans:
            if max(start, s) < min(end, e):
                return True
        return False

    def _record_span(start: int, end: int) -> None:
        occupied_spans.append((start, end))

    # -------------------------------------------------------------
    # 1. Aadhaar: Regex Candidate -> Normalization -> Verhoeff Check
    # -------------------------------------------------------------
    for match in RE_AADHAAR_CANDIDATE.finditer(text):
        raw_match = match.group(0)
        start, end = match.span()

        # Step 2: Normalization
        cleaned_num = clean_numeric_string(raw_match)

        # Step 3: Verhoeff Validation
        if is_valid_aadhaar(cleaned_num):
            detections.append(
                Detection(
                    entity_type="Aadhaar",
                    start=start,
                    end=end,
                    confidence=1.0,
                    detector="regex+verhoeff",
                    valid=True,
                    severity=SeverityLevel.HIGH,
                    explanation="12-digit Aadhaar number with valid Verhoeff checksum.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # -------------------------------------------------------------
    # 2. Card-Like Numbers: Regex Candidate -> Luhn Checksum Check
    # -------------------------------------------------------------
    for match in RE_CARD.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        cleaned_digits = clean_numeric_string(raw_match)

        # Require 13-19 digits and Luhn Mod-10 pass
        if 13 <= len(cleaned_digits) <= 19 and luhn_checksum_validate(cleaned_digits):
            detections.append(
                Detection(
                    entity_type="Card",
                    start=start,
                    end=end,
                    confidence=1.0,
                    detector="regex+luhn",
                    valid=True,
                    severity=SeverityLevel.CRITICAL,
                    explanation="Payment card number validated by Luhn Mod-10 algorithm.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # -------------------------------------------------------------
    # 3. PAN: Structural Regex Validation
    # -------------------------------------------------------------
    for match in RE_PAN.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        if is_valid_pan(raw_match):
            detections.append(
                Detection(
                    entity_type="PAN",
                    start=start,
                    end=end,
                    confidence=0.95,
                    detector="regex_structural",
                    valid=True,
                    severity=SeverityLevel.HIGH,
                    explanation="Indian Permanent Account Number (PAN) matching structural pattern.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # -------------------------------------------------------------
    # 4. IFSC: Indian Financial System Code
    # -------------------------------------------------------------
    for match in RE_IFSC.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        if is_valid_ifsc(raw_match):
            detections.append(
                Detection(
                    entity_type="IFSC",
                    start=start,
                    end=end,
                    confidence=0.95,
                    detector="regex_structural",
                    valid=True,
                    severity=SeverityLevel.MEDIUM,
                    explanation="Indian Financial System Code (IFSC) matching branch pattern.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # -------------------------------------------------------------
    # 5. Email: RFC Address Matching
    # -------------------------------------------------------------
    for match in RE_EMAIL.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        detections.append(
            Detection(
                entity_type="Email",
                start=start,
                end=end,
                confidence=0.95,
                detector="rfc5322_regex",
                valid=True,
                severity=SeverityLevel.LOW,
                explanation="Standard email address identifier.",
                raw_value=raw_match,
            )
        )
        _record_span(start, end)

    # -------------------------------------------------------------
    # 6. IP Addresses: IPv4 Octet Validation
    # -------------------------------------------------------------
    for match in RE_IPV4.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        if is_valid_ipv4(raw_match):
            detections.append(
                Detection(
                    entity_type="IP Address",
                    start=start,
                    end=end,
                    confidence=0.95,
                    detector="regex_ipv4",
                    valid=True,
                    severity=SeverityLevel.MEDIUM,
                    explanation="Valid IPv4 network address.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # -------------------------------------------------------------
    # 7. Indian Phone Numbers
    # -------------------------------------------------------------
    for match in RE_INDIAN_PHONE.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        cleaned_digits = clean_numeric_string(raw_match)

        # Strip country code prefix (+91 / 91 / 0) to extract 10-digit mobile
        if cleaned_digits.startswith("91") and len(cleaned_digits) == 12:
            cleaned_digits = cleaned_digits[2:]
        elif cleaned_digits.startswith("0") and len(cleaned_digits) == 11:
            cleaned_digits = cleaned_digits[1:]

        # Must be exactly 10 digits starting with 6-9, avoiding degenerate repetition
        if len(cleaned_digits) == 10 and cleaned_digits[0] in "6789" and len(set(cleaned_digits)) > 1:
            detections.append(
                Detection(
                    entity_type="Phone",
                    start=start,
                    end=end,
                    confidence=0.90,
                    detector="regex_prefix",
                    valid=True,
                    severity=SeverityLevel.MEDIUM,
                    explanation="10-digit Indian mobile number with valid prefix and formatting.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # -------------------------------------------------------------
    # 8. Bank Account: Requires Financial Context Proximity
    # -------------------------------------------------------------
    # Find all bank context keyword spans
    context_matches = list(RE_BANK_CONTEXT.finditer(text))

    for match in RE_BANK_ACCOUNT_DIGITS.finditer(text):
        start, end = match.span()
        if _is_overlapping(start, end):
            continue

        raw_match = match.group(0)
        # Check if an account context keyword appears within 40 chars before or after the number
        has_context = False
        for ctx in context_matches:
            c_start, c_end = ctx.span()
            if abs(start - c_end) <= 40 or abs(c_start - end) <= 25:
                has_context = True
                break

        if has_context:
            detections.append(
                Detection(
                    entity_type="Bank Account",
                    start=start,
                    end=end,
                    confidence=0.90,
                    detector="contextual_regex",
                    valid=True,
                    severity=SeverityLevel.HIGH,
                    explanation="Bank account number identified in close proximity to financial keywords.",
                    raw_value=raw_match,
                )
            )
            _record_span(start, end)

    # Sort findings by start offset ascending
    detections.sort(key=lambda d: d.start)

    # Never log raw sensitive text: emit only counts and entity categories
    category_summary = [d.entity_type for d in detections]
    logger.info("Deterministic scan complete: identified %d finding(s) %s",
                len(detections), category_summary)

    return detections
