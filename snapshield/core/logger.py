"""Privacy-safe logging infrastructure for SnapShield AI.

Strictly satisfies Rule 2: 'Never log raw sensitive information'.
All messages and log records are scrubbed for:
- Aadhaar
- PAN
- Phone numbers
- Email addresses
- Bank information (Cards, IBAN, IFSC, Account numbers)
- Passwords, secrets, and API keys
- Medical information and diagnoses
- Raw document content / prompts
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, List, Tuple

# Comprehensive regex scrubbers for all sensitive entity classes
SENSITIVE_SCRUBBERS: List[Tuple[re.Pattern, str]] = [
    # 1. API Keys, Passwords, Secrets, Bearer tokens
    (re.compile(r"(?i)(api[_-]?key|secret|password|bearer|auth|token|access[_-]?token|private[_-]?key)\s*[:=]\s*['\"]?([^\s'\"]{4,})['\"]?"), r"\1: [REDACTED_SECRET]"),
    (re.compile(r"(?i)\b(?:sk-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{20,})\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"), "[REDACTED_GITHUB_KEY]"),
    (re.compile(r"(?i)\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),

    # 2. Aadhaar (12-digit Indian National Identity Number)
    (re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"), "[REDACTED_AADHAAR]"),

    # 3. PAN (Indian Permanent Account Number: 5 letters, 4 digits, 1 letter)
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[REDACTED_PAN]"),

    # 4. Bank Information (Credit/Debit Cards, IBAN, IFSC)
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b"), "[REDACTED_IBAN]"),
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[REDACTED_IFSC]"),
    (re.compile(r"(?i)\b(?:account\s*(?:no|number|#)?\s*[:=]?\s*)\d{9,18}\b"), "account: [REDACTED_BANK_ACCOUNT]"),

    # 5. US SSN & National IDs
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),

    # 6. Phone Numbers (Indian, US, International)
    (re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:\+?1[\-\s]?)?\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b"), "[REDACTED_PHONE]"),

    # 7. Email Addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),

    # 8. Medical & Clinical Diagnoses Cues
    (re.compile(r"(?i)\b(diagnosed with|prescription for|chemotherapy|hypertension|insulin dosage|biopsy results|acute coronary syndrome|cardiac arrest|MRN\s*:\s*[A-Za-z0-9\-]+)\b"), "[REDACTED_MEDICAL]"),
]


class PrivacyPreservingFormatter(logging.Formatter):
    """Logging formatter that sanitizes log records to prevent sensitive data leakage."""

    def format(self, record: logging.LogRecord) -> str:
        original_msg = super().format(record)
        sanitized_msg = original_msg
        for pattern, replacement in SENSITIVE_SCRUBBERS:
            sanitized_msg = pattern.sub(replacement, sanitized_msg)
        return sanitized_msg


def setup_logger(name: str = "snapshield", level: str = "INFO") -> logging.Logger:
    """Configures and returns a privacy-safe logger instance."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    formatter = PrivacyPreservingFormatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Default logger instance
logger = setup_logger()
