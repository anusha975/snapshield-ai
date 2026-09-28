"""Authoritative policy decision and enforcement engine for SnapShield AI.

Features:
- Supported policy profiles:
  - Personal (protects PII, financial data, and medical information)
  - Developer (blocks API keys, passwords, and tokens while allowing code and non-sensitive technical data)
  - Medical (protects health records, patient identifiers, and clinical findings)
- Authoritative Actions:
  - ALLOW: Permitted for outbound forwarding without sanitization.
  - REDACT: Sensitive entity must be redacted or masked prior to transmission.
  - BLOCK: Critical violation; document or egress is outright rejected.
  - REVIEW: Human-in-the-loop manual inspection required before release.
- Loads policy definitions centrally from config/policies.json.
- Invariant: Unknown sensitive categories strictly default to REVIEW.
- Invariant: Every detection receives a decision with a human-readable justification reason.
- Never prints or exposes raw sensitive text in decisions or logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from snapshield.config.policies import (
    PoliciesConfig,
    PolicyConfigManager,
    get_policy_manager,
    normalize_category_key,
)
from snapshield.core.logger import logger
from snapshield.core.models import Detection, PolicyAction, PolicyDecision


# Canonical profile names supported by the policy subsystem
SUPPORTED_MODES: List[str] = ["Personal", "Developer", "Medical"]


def _format_decision_reason(
    mode: str,
    entity_type: str,
    action: PolicyAction,
    is_explicit_rule: bool,
) -> str:
    """Generates a human-readable, auditable justification for a policy decision."""
    if not is_explicit_rule:
        return (
            f"Category '{entity_type}' is not explicitly defined in the '{mode}' policy profile; "
            f"strictly defaulting to {action.value} for privacy preservation."
        )

    if action == PolicyAction.BLOCK:
        return (
            f"Blocked by '{mode}' policy: '{entity_type}' is classified as high-risk sensitive data "
            f"that cannot be shared with external services."
        )
    elif action == PolicyAction.REDACT:
        return (
            f"Redacted under '{mode}' policy: '{entity_type}' contains private identifiable information "
            f"and must be sanitized before transmission."
        )
    elif action == PolicyAction.ALLOW:
        return (
            f"Permitted by '{mode}' policy: '{entity_type}' is approved for standard outbound transmission."
        )
    else:  # REVIEW
        return (
            f"Manual review required under '{mode}' policy: '{entity_type}' requires human authorization "
            f"prior to egress."
        )


class PolicyEngine:
    """Evaluates detections against policy profiles defined in config/policies.json."""

    def __init__(
        self,
        mode: str = "Personal",
        policy_manager: Optional[PolicyConfigManager] = None,
    ) -> None:
        self.policy_manager = policy_manager or get_policy_manager()
        self.mode = self._resolve_mode_name(mode)

    def _resolve_mode_name(self, requested_mode: str) -> str:
        """Resolves requested profile name against available profiles (case-insensitive)."""
        available = self.policy_manager.list_profiles()
        for name in available:
            if name.lower() == requested_mode.strip().lower():
                return name

        logger.warning(
            "Requested policy mode '%s' not found in %s; defaulting to '%s'.",
            requested_mode,
            available,
            available[0] if available else "Personal",
        )
        return available[0] if available else "Personal"

    def evaluate_detection(self, detection: Union[Detection, Any]) -> PolicyDecision:
        """Evaluates a single detection and returns a strongly typed PolicyDecision."""
        # Safely extract entity_type string
        raw_type = (
            getattr(detection, "entity_type", None)
            or getattr(detection, "type", None)
            or getattr(detection, "entity_category", "Unknown")
        )
        if hasattr(raw_type, "value"):
            raw_type = raw_type.value
        entity_type_str = str(raw_type)

        profile = self.policy_manager.config.get_profile(self.mode)
        normalized_key = normalize_category_key(entity_type_str)

        # Check if rule exists explicitly in profile
        is_explicit = normalized_key in profile._normalized_rules
        action = profile.resolve_action(entity_type_str, default_action=self.policy_manager.config.default_action)

        reason = _format_decision_reason(
            mode=self.mode,
            entity_type=entity_type_str,
            action=action,
            is_explicit_rule=is_explicit,
        )

        return PolicyDecision(
            entity_type=entity_type_str,
            action=action,
            reason=reason,
            profile_applied=self.mode,
        )

    def apply_policy(
        self,
        detections: List[Union[Detection, Any]],
    ) -> List[PolicyDecision]:
        """Applies policy to every detection in the list.

        Guarantees:
        - Every detection receives a corresponding decision (1:1 mapped).
        - Every decision contains a human-readable reason.
        - Unknown sensitive categories default to REVIEW.
        """
        if not detections:
            return []

        decisions: List[PolicyDecision] = []
        for det in detections:
            decision = self.evaluate_detection(det)
            decisions.append(decision)

        logger.info(
            "PolicyEngine applied '%s' profile to %d detection(s). Decisions: %s",
            self.mode,
            len(decisions),
            [d.action.value for d in decisions],
        )
        return decisions

    def get_aggregate_decision(
        self,
        decisions: List[PolicyDecision],
    ) -> PolicyDecision:
        """Computes aggregate document-level policy decision based on precedence:

        BLOCK > REVIEW > REDACT > ALLOW.
        """
        if not decisions:
            return PolicyDecision(
                entity_type="*",
                action=PolicyAction.ALLOW,
                reason=f"No sensitive entities found under '{self.mode}' policy profile.",
                profile_applied=self.mode,
            )

        actions = [d.action for d in decisions]

        if PolicyAction.BLOCK in actions:
            block_entities = [d.entity_type for d in decisions if d.action == PolicyAction.BLOCK]
            return PolicyDecision(
                entity_type="*",
                action=PolicyAction.BLOCK,
                reason=f"Document blocked: Contains blocked sensitive entity/entities {list(set(block_entities))}.",
                profile_applied=self.mode,
            )

        if PolicyAction.REVIEW in actions:
            review_entities = [d.entity_type for d in decisions if d.action == PolicyAction.REVIEW]
            return PolicyDecision(
                entity_type="*",
                action=PolicyAction.REVIEW,
                reason=f"Document requires manual review: Contains unverified entity/entities {list(set(review_entities))}.",
                profile_applied=self.mode,
            )

        if PolicyAction.REDACT in actions:
            redact_entities = [d.entity_type for d in decisions if d.action == PolicyAction.REDACT]
            return PolicyDecision(
                entity_type="*",
                action=PolicyAction.REDACT,
                reason=f"Document safe to forward after redacting sensitive entity/entities {list(set(redact_entities))}.",
                profile_applied=self.mode,
            )

        return PolicyDecision(
            entity_type="*",
            action=PolicyAction.ALLOW,
            reason=f"All detected entities permitted for outbound transfer under '{self.mode}' policy.",
            profile_applied=self.mode,
        )


def apply_policy(
    detections: List[Union[Detection, Any]],
    mode: str = "Personal",
) -> List[PolicyDecision]:
    """Applies policy profile to a list of detections.

    Args:
        detections: List of Detection or Finding objects to evaluate.
        mode: Policy mode ('Personal', 'Developer', or 'Medical'). Defaults to 'Personal'.

    Returns:
        List of strongly typed PolicyDecision objects, one for each input detection.

    Guarantees:
        - Every detection receives a decision.
        - Every decision contains a human-readable reason.
        - Unknown sensitive categories default to PolicyAction.REVIEW.
    """
    engine = PolicyEngine(mode=mode)
    return engine.apply_policy(detections)
