"""Redaction and Sanitization Engine for SnapShield AI.

Enforces:
- Rule 1: Never send original sensitive content to an external LLM.
- Rule 8: Keep original input separate from sanitized output.
"""

from __future__ import annotations

import hashlib
from typing import List
from snapshield.core.exceptions import RedactionError
from snapshield.core.types import Finding, RedactionResult


class RedactionEngine:
    """Sanitizes text by replacing detected sensitive entity spans with safe tokens."""

    @staticmethod
    def redact_text(original_text: str, findings: List[Finding]) -> RedactionResult:
        """Applies masking placeholders to all findings without altering the original input string.

        Processes findings in reverse positional order to maintain valid string slice offsets.
        """
        if not original_text:
            return RedactionResult(
                sanitized_text="",
                redactions_applied=0,
                findings_redacted=[],
                integrity_hash=hashlib.sha256(b"").hexdigest(),
            )

        if not findings:
            return RedactionResult(
                sanitized_text=original_text,
                redactions_applied=0,
                findings_redacted=[],
                integrity_hash=hashlib.sha256(original_text.encode("utf-8")).hexdigest(),
            )

        try:
            # Sort findings by start_pos descending to preserve index validity during replacement
            sorted_findings = sorted(findings, key=lambda f: f.start_pos, reverse=True)

            # Build sanitized text from original text
            chars = list(original_text)
            applied_count = 0
            applied_findings: List[Finding] = []

            for finding in sorted_findings:
                start = finding.start_pos
                end = finding.end_pos

                if 0 <= start <= end <= len(chars):
                    chars[start:end] = list(finding.mask_placeholder)
                    applied_count += 1
                    applied_findings.append(finding)

            sanitized_str = "".join(chars)
            integrity_hash = hashlib.sha256(sanitized_str.encode("utf-8")).hexdigest()

            return RedactionResult(
                sanitized_text=sanitized_str,
                redactions_applied=applied_count,
                findings_redacted=applied_findings,
                integrity_hash=integrity_hash,
            )

        except Exception as exc:
            raise RedactionError(f"Failed to redact sensitive content: {exc}") from exc
