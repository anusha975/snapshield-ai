"""Redaction and content sanitization engine for SnapShield AI.

Features:
- Enforces policy-guided content transformations:
  - REDACT: Replaces sensitive span with [REDACTED: ENTITY_TYPE].
  - ALLOW: Preserves original text without alteration.
  - BLOCK: Marks content as prohibited from external transmission.
  - REVIEW: Flags content as requiring explicit user approval.
- Enforces architectural invariants:
  1. Never overwrites or mutates original input text.
  2. Performs replacements strictly from the end of the text toward the beginning
     to prevent character index shifting and corruption.
  3. Verifies that blocked or redacted values cannot accidentally remain in sanitized text.
  4. Resolves adjacent and overlapping detection spans safely.
- Returns strongly typed RedactionResult with:
  - sanitized_text
  - redaction_count
  - summary
  - blocked
  - requires_review
  - integrity_hash
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from snapshield.config.policies import normalize_category_key
from snapshield.core.exceptions import RedactionError
from snapshield.core.logger import logger
from snapshield.core.models import Detection, PolicyAction, PolicyDecision
from snapshield.core.policy import apply_policy


@dataclass
class RedactionResult:
    """Strongly typed result of a sanitization operation.

    Supports both attribute access and 5-tuple unpacking:
    sanitized_text, redaction_count, summary, blocked, requires_review = result
    """
    sanitized_text: str
    redaction_count: int
    summary: str
    blocked: bool
    requires_review: bool
    integrity_hash: str = ""
    redacted_entities: List[str] = field(default_factory=list)

    def __iter__(self):
        """Allows unpacking as a 5-tuple matching user specification."""
        return iter((
            self.sanitized_text,
            self.redaction_count,
            self.summary,
            self.blocked,
            self.requires_review,
        ))

    def safe_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary safe for logging and telemetry."""
        return {
            "redaction_count": self.redaction_count,
            "blocked": self.blocked,
            "requires_review": self.requires_review,
            "summary": self.summary,
            "integrity_hash": self.integrity_hash,
            "redacted_entities": self.redacted_entities,
        }


def format_placeholder(entity_type: str, action: PolicyAction) -> str:
    """Constructs the standard replacement token for an entity.

    REDACT: [REDACTED: ENTITY_TYPE]
    BLOCK:  [BLOCKED: ENTITY_TYPE]
    REVIEW: [REVIEW_REQUIRED: ENTITY_TYPE]
    """
    clean_type = normalize_category_key(entity_type)
    if action == PolicyAction.BLOCK:
        return f"[BLOCKED: {clean_type}]"
    elif action == PolicyAction.REVIEW:
        return f"[REDACTED: {clean_type}]"
    return f"[REDACTED: {clean_type}]"


def _consolidate_replacement_spans(
    raw_spans: List[Tuple[int, int, str, str]],  # (start, end, placeholder, raw_value)
) -> List[Tuple[int, int, str, List[str]]]:
    """Resolves overlapping spans prior to reverse replacement.

    Adjacent spans (where end1 == start2) are preserved independently.
    Overlapping spans (where start2 < end1) are merged into a single safe union span.
    """
    if not raw_spans:
        return []

    # Sort ascending by start position, then end position
    sorted_spans = sorted(raw_spans, key=lambda s: (s[0], s[1]))
    merged: List[Tuple[int, int, str, List[str]]] = []

    for start, end, placeholder, raw_val in sorted_spans:
        if not merged:
            merged.append((start, end, placeholder, [raw_val] if raw_val else []))
            continue

        prev_start, prev_end, prev_placeholder, prev_raws = merged[-1]

        # Check for overlap (start < prev_end)
        if start < prev_end:
            # Union span covering both overlapping detections
            new_end = max(prev_end, end)
            # Combine placeholders if different
            combined_placeholder = prev_placeholder if prev_placeholder == placeholder else "[REDACTED: SENSITIVE_DATA]"
            combined_raws = list(prev_raws)
            if raw_val and raw_val not in combined_raws:
                combined_raws.append(raw_val)

            merged[-1] = (prev_start, new_end, combined_placeholder, combined_raws)
        else:
            # Adjacent or disjoint span
            merged.append((start, end, placeholder, [raw_val] if raw_val else []))

    return merged


def verify_sanitization_leak_free(
    sanitized_text: str,
    sensitive_values_to_verify: Set[str],
) -> None:
    """Verifies that none of the raw sensitive values accidentally remain in the sanitized text.

    Raises RedactionError if any raw sensitive value persists in non-placeholder text.
    """
    # Strip formal placeholders so category names (e.g. 'CONFIDENTIAL') inside tokens
    # like [REDACTED: CONFIDENTIAL_BUSINESS_INFORMATION] do not cause false-positive leak alarms
    content_outside_placeholders = re.sub(
        r"\[(?:REDACTED|BLOCKED|REVIEW_REQUIRED):\s*[^\]]+\]", "", sanitized_text
    )

    for raw in sensitive_values_to_verify:
        clean_raw = raw.strip()
        # Avoid checking trivial whitespace or sub-3-char fragments
        if len(clean_raw) >= 3:
            if clean_raw in content_outside_placeholders:
                logger.error("Sanitization leak verification failed! A sensitive token was found in sanitized text.")
                raise RedactionError(
                    "Sanitization verification failed: A blocked or redacted sensitive value "
                    "accidentally remained in the output text."
                )


class Redactor:
    """Engine responsible for safe content sanitization."""

    @staticmethod
    def redact(
        original_text: str,
        detections: List[Union[Detection, Any]],
        decisions: Optional[List[PolicyDecision]] = None,
        mode: str = "Personal",
    ) -> RedactionResult:
        """Sanitizes text based on detections and policy decisions.

        Args:
            original_text: The original string. Never mutated or overwritten.
            detections: List of detected entities.
            decisions: Optional policy decisions corresponding to detections.
                       If None, apply_policy(detections, mode) is automatically run.
            mode: Policy mode ('Personal', 'Developer', 'Medical').

        Returns:
            RedactionResult containing sanitized_text, redaction_count, summary,
            blocked flag, requires_review flag, and SHA-256 integrity hash.
        """
        if not original_text or not isinstance(original_text, str):
            empty_hash = hashlib.sha256(b"").hexdigest()
            return RedactionResult(
                sanitized_text="",
                redaction_count=0,
                summary="Empty or non-string input provided.",
                blocked=False,
                requires_review=False,
                integrity_hash=empty_hash,
            )

        if not detections:
            text_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
            return RedactionResult(
                sanitized_text=original_text,
                redaction_count=0,
                summary="No sensitive entities detected; content preserved as-is.",
                blocked=False,
                requires_review=False,
                integrity_hash=text_hash,
            )

        # 1. Resolve decisions for each detection
        if decisions is None or len(decisions) != len(detections):
            decisions = apply_policy(detections, mode=mode)

        # 2. Inspect decisions to establish overall blocked and review flags
        is_blocked = False
        requires_review = False
        blocked_types: List[str] = []
        review_types: List[str] = []
        redacted_types: List[str] = []

        spans_to_replace: List[Tuple[int, int, str, str]] = []
        sensitive_values_to_verify: Set[str] = set()

        for det, dec in zip(detections, decisions):
            entity_type = getattr(det, "entity_type", None) or getattr(det, "type", "UNKNOWN")
            start = getattr(det, "start", 0)
            end = getattr(det, "end", 0)
            raw_val = getattr(det, "raw_value", None) or original_text[start:end]

            if dec.action == PolicyAction.BLOCK:
                is_blocked = True
                blocked_types.append(entity_type)
                placeholder = format_placeholder(entity_type, PolicyAction.BLOCK)
                spans_to_replace.append((start, end, placeholder, raw_val))
                if raw_val:
                    sensitive_values_to_verify.add(raw_val)

            elif dec.action == PolicyAction.REVIEW:
                requires_review = True
                review_types.append(entity_type)
                placeholder = format_placeholder(entity_type, PolicyAction.REVIEW)
                spans_to_replace.append((start, end, placeholder, raw_val))
                if raw_val:
                    sensitive_values_to_verify.add(raw_val)

            elif dec.action == PolicyAction.REDACT:
                redacted_types.append(entity_type)
                placeholder = format_placeholder(entity_type, PolicyAction.REDACT)
                spans_to_replace.append((start, end, placeholder, raw_val))
                if raw_val:
                    sensitive_values_to_verify.add(raw_val)

            elif dec.action == PolicyAction.ALLOW:
                # ALLOW: Preserve original text (do not replace)
                pass

        # 3. Consolidate overlapping spans to prevent index corruption
        consolidated_spans = _consolidate_replacement_spans(spans_to_replace)

        # 4. Perform replacements from the END of the text toward the BEGINNING
        # Sorting descending by start position guarantees preceding offsets remain valid.
        consolidated_spans.sort(key=lambda s: s[0], reverse=True)

        text_chars = list(original_text)
        applied_count = 0

        for start, end, placeholder, _ in consolidated_spans:
            # Bound check
            if 0 <= start <= end <= len(text_chars):
                text_chars[start:end] = list(placeholder)
                applied_count += 1

        sanitized_str = "".join(text_chars)

        # 5. Safety Verification: Ensure blocked/redacted raw values cannot accidentally remain
        verify_sanitization_leak_free(sanitized_str, sensitive_values_to_verify)

        # 6. Build summary description
        integrity_hash = hashlib.sha256(sanitized_str.encode("utf-8")).hexdigest()

        summary_parts: List[str] = []
        if is_blocked:
            summary_parts.append(
                f"PROHIBITED: Content contains blocked entity/entities {list(set(blocked_types))} "
                f"and is strictly prohibited from external transmission."
            )
        if requires_review:
            summary_parts.append(
                f"REVIEW REQUIRED: Entities {list(set(review_types))} require explicit user approval."
            )
        if redacted_types:
            summary_parts.append(
                f"REDACTED: Replaced {len(redacted_types)} sensitive finding(s) {list(set(redacted_types))} with safe tokens."
            )
        if not summary_parts:
            summary_parts.append("All detected entities permitted under active policy.")

        full_summary = " | ".join(summary_parts)

        logger.info(
            "Redaction complete: %d replacements applied (blocked=%s, review=%s). Hash=%s",
            applied_count,
            is_blocked,
            requires_review,
            integrity_hash[:8],
        )

        return RedactionResult(
            sanitized_text=sanitized_str,
            redaction_count=applied_count,
            summary=full_summary,
            blocked=is_blocked,
            requires_review=requires_review,
            integrity_hash=integrity_hash,
            redacted_entities=list(set(redacted_types + blocked_types + review_types)),
        )


# Module-level convenience functions
def redact_text(
    original_text: str,
    detections: List[Union[Detection, Any]],
    decisions: Optional[List[PolicyDecision]] = None,
    mode: str = "Personal",
) -> RedactionResult:
    """Sanitizes text by replacing sensitive spans from end toward beginning."""
    return Redactor.redact(
        original_text=original_text,
        detections=detections,
        decisions=decisions,
        mode=mode,
    )


# Alias function matching common naming
def redact(
    original_text: str,
    detections: List[Union[Detection, Any]],
    decisions: Optional[List[PolicyDecision]] = None,
    mode: str = "Personal",
) -> RedactionResult:
    """Convenience alias for redact_text."""
    return redact_text(original_text, detections, decisions, mode=mode)
