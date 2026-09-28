"""Result merger and multi-engine consolidation module for SnapShield AI.

Combines:
1. Deterministic structured detection (Aadhaar, PAN, phone numbers, email, etc.)
2. Deterministic secret & credential detection (API keys, passwords, JWT, AWS, DB URIs, etc.)
3. Local AI contextual detection (PERSON, ADDRESS, ORGANIZATION, MEDICAL_INFORMATION, etc.)

Guarantees:
- Merges detections across engines.
- Removes exact duplicate detections.
- Resolves overlapping spans between detectors of the same entity type.
- Preserves different entity types when spans overlap or are co-located (e.g. Phone + Person).
- Preserves detector sources, recording multi-detector corroboration.
- Combines and preserves detection confidence mathematically.
- Returns detections sorted deterministically by character start position.
- Never prints or exposes raw secret values in logs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from snapshield.config.policies import normalize_category_key
from snapshield.core.ai_detector import AIDetector, LocalAIDetector
from snapshield.core.code_detectors import detect_credentials
from snapshield.core.detectors import detect_structured
from snapshield.core.logger import logger
from snapshield.core.models import Detection, SeverityLevel


# Qualitative severity hierarchy for conflict resolution
SEVERITY_ORDER: Dict[SeverityLevel, int] = {
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}

_GLOBAL_AI_DETECTOR: Optional[AIDetector] = None


def get_default_ai_detector() -> AIDetector:
    """Returns or creates a shared LocalAIDetector instance."""
    global _GLOBAL_AI_DETECTOR
    if _GLOBAL_AI_DETECTOR is None:
        _GLOBAL_AI_DETECTOR = LocalAIDetector(allow_dev_fallback=True)
    return _GLOBAL_AI_DETECTOR


def _merge_two_detections(
    d1: Detection,
    d2: Detection,
    text: Optional[str] = None,
) -> Detection:
    """Merges two detections of the same entity type into a single corroborated Detection.

    - Span: Expanded bounding span covering both observations.
    - Detector: Combined list of contributing detectors.
    - Confidence: Noisy-OR evidence combination (confidence increases on corroboration).
    - Severity: Highest severity classification between the two.
    - Explanation: Corroboration summary preserving diagnostic details.
    """
    new_start = min(d1.start, d2.start)
    new_end = max(d1.end, d2.end)

    # Combine unique detector sources
    sources: List[str] = []
    for s in d1.detector.split(" + "):
        clean_s = s.strip()
        if clean_s and clean_s not in sources:
            sources.append(clean_s)
    for s in d2.detector.split(" + "):
        clean_s = s.strip()
        if clean_s and clean_s not in sources:
            sources.append(clean_s)
    combined_detector = " + ".join(sources)

    # Combine confidence via Noisy-OR evidence accumulation
    c1 = float(d1.confidence)
    c2 = float(d2.confidence)
    noisy_or_conf = min(1.0, round(1.0 - (1.0 - c1) * (1.0 - c2), 4))
    combined_confidence = max(noisy_or_conf, c1, c2)

    # Select highest severity
    s1_val = SEVERITY_ORDER.get(d1.severity, 0)
    s2_val = SEVERITY_ORDER.get(d2.severity, 0)
    merged_severity = d1.severity if s1_val >= s2_val else d2.severity

    # Prefer d1 entity type if c1 >= c2, else d2
    merged_entity_type = d1.entity_type if c1 >= c2 else d2.entity_type

    # Explanation
    if len(sources) > 1:
        merged_explanation = (
            f"Corroborated by multiple detectors ({combined_detector}): "
            f"{d1.explanation} | {d2.explanation}"
        )
    else:
        merged_explanation = d1.explanation if c1 >= c2 else d2.explanation

    # Raw value derivation
    if text is not None and 0 <= new_start <= len(text) and 0 <= new_end <= len(text):
        raw_val = text[new_start:new_end]
    elif d1.start == new_start and d1.end == new_end:
        raw_val = d1.raw_value
    elif d2.start == new_start and d2.end == new_end:
        raw_val = d2.raw_value
    else:
        raw_val = d1.raw_value or d2.raw_value

    return Detection(
        entity_type=merged_entity_type,
        start=new_start,
        end=new_end,
        confidence=combined_confidence,
        detector=combined_detector,
        valid=d1.valid and d2.valid,
        severity=merged_severity,
        explanation=merged_explanation,
        raw_value=raw_val,
    )


def merge_detections(
    detections: List[Detection],
    text: Optional[str] = None,
) -> List[Detection]:
    """Merges a collection of detections from multiple detector subsystems.

    Guarantees:
    1. Removes exact duplicates (same span & same entity type).
    2. Merges overlapping spans of the SAME entity type with evidence accumulation.
    3. Preserves different entity types even when spans overlap or are co-located
       (e.g. Phone and Person both remain).
    4. Preserves detector sources and confidence.
    5. Returns detections strictly sorted by character start position.
    """
    if not detections:
        return []

    # 1. Group detections by normalized entity category
    # Detections of DIFFERENT entity types are preserved independently.
    type_groups: Dict[str, List[Detection]] = {}
    for d in detections:
        norm_key = normalize_category_key(d.entity_type)
        type_groups.setdefault(norm_key, []).append(d)

    merged_results: List[Detection] = []

    # 2. Within each entity category, merge overlapping spans and duplicate evidence
    for norm_key, group in type_groups.items():
        # Sort by start position, then end position
        sorted_group = sorted(group, key=lambda d: (d.start, d.end))

        merged_group: List[Detection] = []
        for det in sorted_group:
            if not merged_group:
                merged_group.append(det)
                continue

            last = merged_group[-1]

            # Overlap condition for SAME entity category:
            # Spans intersect if max(start1, start2) <= min(end1, end2)
            is_overlapping = max(last.start, det.start) <= min(last.end, det.end)

            if is_overlapping:
                # Merge the duplicate / overlapping evidence
                merged_det = _merge_two_detections(last, det, text)
                merged_group[-1] = merged_det
            else:
                merged_group.append(det)

        merged_results.extend(merged_group)

    # 3. Sort all merged results by position
    merged_results.sort(key=lambda d: (d.start, d.end))

    # Log summary without exposing raw secret contents
    logger.info(
        "ResultMerger: consolidated %d raw detections into %d unified detections.",
        len(detections),
        len(merged_results),
    )

    return merged_results


def scan_text(
    text: str,
    ai_enabled: bool = True,
    ai_detector: Optional[AIDetector] = None,
) -> List[Detection]:
    """Executes a unified multi-engine sensitivity scan over input text.

    Combines:
    1. Deterministic Structured Detection (Aadhaar, PAN, Phone, Email, Bank Account, IFSC, Cards, IP)
    2. Deterministic Secret & Credential Detection (API keys, passwords, JWT, AWS, URLs, DB URIs)
    3. Local Contextual AI Detection (PERSON, ADDRESS, ORGANIZATION, MEDICAL_INFORMATION, etc.)

    Args:
        text: The raw text to inspect.
        ai_enabled: Whether to run the local AI contextual detector. Defaults to True.
        ai_detector: Optional explicit AIDetector instance (uses shared LocalAIDetector by default).

    Returns:
        Unified, deduplicated, and position-sorted list of Detection objects.
    """
    if not text or not isinstance(text, str):
        return []

    collected_detections: List[Detection] = []

    # 1. Deterministic Structured Detection
    try:
        struct_findings = detect_structured(text)
        collected_detections.extend(struct_findings)
    except Exception as exc:
        logger.error("Structured detector failed during scan_text: %s", str(exc))

    # 2. Deterministic Secret & Credential Detection
    try:
        cred_findings = detect_credentials(text)
        collected_detections.extend(cred_findings)
    except Exception as exc:
        logger.error("Credential detector failed during scan_text: %s", str(exc))

    # 3. Local AI Contextual Detection (if enabled)
    if ai_enabled:
        try:
            active_ai = ai_detector or get_default_ai_detector()
            if active_ai.is_available:
                ai_findings = active_ai.detect(text)
                collected_detections.extend(ai_findings)
            else:
                logger.info("Local AI detector is unavailable; skipped contextual scan.")
        except Exception as exc:
            logger.error("Local AI detector failed during scan_text: %s", str(exc))

    # 4. Merge, deduplicate, and resolve overlaps
    unified = merge_detections(collected_detections, text=text)
    return unified


class ResultMerger:
    """Class wrapper providing stateful or configured result merging services."""

    def __init__(self, ai_detector: Optional[AIDetector] = None) -> None:
        self.ai_detector = ai_detector

    def merge(
        self,
        detections: List[Detection],
        text: Optional[str] = None,
    ) -> List[Detection]:
        """Merges arbitrary detection collections."""
        return merge_detections(detections, text=text)

    def scan_text(self, text: str, ai_enabled: bool = True) -> List[Detection]:
        """Executes full scan pipeline using the configured detector."""
        return scan_text(text, ai_enabled=ai_enabled, ai_detector=self.ai_detector)
