"""Privacy Policy Decision Service for SnapShield AI.

Translates detected entities and risk assessments into authoritative policy decisions:
BLOCK, REDACT, REVIEW, or ALLOW, backed by configurable policy profiles (Personal, Developer).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from snapshield.config.policies import (
    PolicyAction,
    PolicyConfigManager,
    get_policy_manager,
)
from snapshield.core.logger import logger
from snapshield.core.types import Finding, PolicyDecision, RiskAssessment


class PolicyEngine:
    """Enforces configurable privacy profiles over evaluated content.

    Business rules are defined centrally in config/policies.json and accessed via PolicyConfigManager.
    Unknown sensitive categories strictly default to PolicyAction.REVIEW.
    """

    def __init__(
        self,
        profile_name: str = "Personal",
        policy_manager: Optional[PolicyConfigManager] = None,
    ) -> None:
        self.policy_manager = policy_manager or get_policy_manager()
        self.profile_name = profile_name

        # Verify profile exists or fallback to available profile
        available = self.policy_manager.list_profiles()
        if self.profile_name not in available:
            logger.warning("Profile '%s' not found in %s; using '%s'", self.profile_name, available, available[0])
            self.profile_name = available[0]

    def evaluate_findings(self, findings: List[Finding]) -> tuple[PolicyAction, Dict[str, PolicyAction], List[str]]:
        """Evaluates a list of findings against the active policy profile.

        Rules precedence:
        1. If any finding triggers BLOCK -> BLOCK
        2. Else if any finding triggers REVIEW (or is an unknown category) -> REVIEW
        3. Else if any finding triggers REDACT -> REDACT
        4. Else (all ALLOW or no findings) -> ALLOW
        """
        if not findings:
            return PolicyAction.ALLOW, {}, ["No sensitive findings detected; content is safe for immediate processing."]

        category_actions: Dict[str, PolicyAction] = {}
        action_counts = {PolicyAction.BLOCK: 0, PolicyAction.REVIEW: 0, PolicyAction.REDACT: 0, PolicyAction.ALLOW: 0}
        reasons: List[str] = []

        for finding in findings:
            category_key = finding.entity_type or finding.entity_category.value
            action = self.policy_manager.get_action_for_category(self.profile_name, category_key)
            category_actions[category_key] = action
            action_counts[action] += 1

        if action_counts[PolicyAction.BLOCK] > 0:
            final_action = PolicyAction.BLOCK
            reasons.append(f"Blocked by policy '{self.profile_name}' due to high-risk entities.")
        elif action_counts[PolicyAction.REVIEW] > 0:
            final_action = PolicyAction.REVIEW
            reasons.append(f"Policy '{self.profile_name}' requires manual review for sensitive/unclassified entities.")
        elif action_counts[PolicyAction.REDACT] > 0:
            final_action = PolicyAction.REDACT
            reasons.append(f"Policy '{self.profile_name}' requires redaction before safe egress.")
        else:
            final_action = PolicyAction.ALLOW
            reasons.append(f"All detected entities are explicitly allowed by '{self.profile_name}' policy.")

        return final_action, category_actions, reasons

    def decide(self, assessment: RiskAssessment, findings: Optional[List[Finding]] = None) -> PolicyDecision:
        """Determines the policy action based on findings and risk assessment."""
        now_utc = datetime.now(timezone.utc).isoformat()

        if findings is not None:
            action, _, reasons = self.evaluate_findings(findings)
            rationale = " ".join(reasons)
        else:
            # Fallback evaluation based on categories in assessment
            score = assessment.total_score
            if "CREDENTIALS" in assessment.category_breakdown and self.profile_name.lower() == "developer":
                action = PolicyAction.BLOCK
                rationale = "Direct credentials / secret keys detected. Outright egress blocked by Developer policy."
            elif score >= 75.0:
                action = PolicyAction.BLOCK
                rationale = f"Risk score ({score}) exceeds blocking threshold."
            elif score >= 45.0:
                action = PolicyAction.REVIEW
                rationale = f"Risk score ({score}) warrants manual security review."
            elif score > 0.0:
                action = PolicyAction.REDACT
                rationale = f"Risk score ({score}) requires sanitization/redaction before egress."
            else:
                action = PolicyAction.ALLOW
                rationale = "No risk detected. Safe for egress."

        return PolicyDecision(
            action=action,
            profile_applied=self.profile_name,
            risk_assessment=assessment,
            enforced_at_utc=now_utc,
            rationale=rationale,
        )
