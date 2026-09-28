"""Hybrid Detection Engine for SnapShield AI.

Combines:
- Deterministic regex pattern matching for structured sensitive data.
- Checksum verification algorithms (e.g. Luhn algorithm for credit card numbers).
- Local contextual detection hooks.
"""

from __future__ import annotations

import re
from typing import List, Optional
from snapshield.core.types import DetectionSource, EntityCategory, Finding
from snapshield.core.logger import logger


def luhn_checksum_validate(number_str: str) -> bool:
    """Validates credit card numbers using the standard Luhn algorithm (Mod-10)."""
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


class HybridDetectionEngine:
    """Analyzes text for sensitive entities using deterministic and algorithmic checks."""

    # Deterministic regex patterns
    PATTERNS = {
        "CREDENTIAL_OPENAI_KEY": (
            re.compile(r"\bsk-[A-Za-z0-9-_]{20,48}\b"),
            EntityCategory.CREDENTIALS,
            "[REDACTED_API_KEY]",
        ),
        "CREDENTIAL_GENERIC_SECRET": (
            re.compile(r"(?i)\b(?:api[_-]?key|secret|auth_token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{16,64})['\"]?\b"),
            EntityCategory.CREDENTIALS,
            "[REDACTED_SECRET]",
        ),
        "CREDENTIAL_PRIVATE_KEY": (
            re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
            EntityCategory.CREDENTIALS,
            "[REDACTED_PRIVATE_KEY]",
        ),
        "PAYMENT_CARD": (
            re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            EntityCategory.PAYMENT_CARD,
            "[REDACTED_PAYMENT_CARD]",
        ),
        "GOVERNMENT_ID_US_SSN": (
            re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
            EntityCategory.GOVERNMENT_ID,
            "[REDACTED_SSN]",
        ),
        "CONTACT_EMAIL": (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            EntityCategory.CONTACT_INFO,
            "[REDACTED_EMAIL]",
        ),
        "CONTACT_PHONE": (
            re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            EntityCategory.CONTACT_INFO,
            "[REDACTED_PHONE]",
        ),
        "INTERNAL_IPV4": (
            re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"),
            EntityCategory.INTERNAL_INFRA,
            "[REDACTED_INTERNAL_IP]",
        ),
    }

    def scan_text(self, text: str) -> List[Finding]:
        """Scans input text and returns structured findings.

        Does not store or leak raw sensitive text.
        """
        findings: List[Finding] = []

        for entity_type, (pattern, category, placeholder) in self.PATTERNS.items():
            for match in pattern.finditer(text):
                matched_str = match.group(0)
                start, end = match.span()

                # Perform checksum validation when applicable
                checksum_ok: Optional[bool] = None
                source = DetectionSource.DETERMINISTIC
                confidence = 0.95

                if category == EntityCategory.PAYMENT_CARD:
                    checksum_ok = luhn_checksum_validate(matched_str)
                    source = DetectionSource.CHECKSUM
                    confidence = 1.0 if checksum_ok else 0.40

                findings.append(
                    Finding(
                        entity_category=category,
                        entity_type=entity_type,
                        start_pos=start,
                        end_pos=end,
                        detection_source=source,
                        confidence_score=confidence,
                        checksum_verified=checksum_ok,
                        mask_placeholder=placeholder,
                    )
                )

        logger.info("Hybrid detection scan complete. Identified %d findings.", len(findings))
        return findings

    def scan_detections(self, text: str) -> list[Detection]:
        """Scans input text and returns unified strongly-typed Detection domain models.

        Integrates both:
        1. Deterministic code and credential detection (API keys, passwords, JWT, AWS, Bearer, DB URIs, etc.)
        2. Structured PII detection (Aadhaar, PAN, phone numbers, email, bank accounts, IFSC, cards, IPv4)

        Enforces:
        - Deterministic pattern detection (zero LLM calls).
        - Non-overlapping spans with priority given to credential detectors.
        - HIGH or CRITICAL severity for credentials.
        - Privacy preservation (never logs raw secret content).
        """
        from snapshield.core.code_detectors import detect_credentials
        from snapshield.core.detectors import detect_structured
        from snapshield.core.models import Detection

        cred_detections = detect_credentials(text)
        struct_detections = detect_structured(text)

        # Merge with priority to credentials on overlapping spans
        merged: list[Detection] = list(cred_detections)
        occupied_spans = [(d.start, d.end) for d in cred_detections]

        for det in struct_detections:
            overlap = any(max(det.start, s) < min(det.end, e) for s, e in occupied_spans)
            if not overlap:
                merged.append(det)
                occupied_spans.append((det.start, det.end))

        merged.sort(key=lambda d: d.start)
        return merged

