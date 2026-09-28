"""External LLM Gatekeeper Service for SnapShield AI.

Strictly enforces:
- Rule 1: Never send original sensitive content to an external LLM.
- Rule 8: Keep original input separate from sanitized output.
- Rule 15: Do not hardcode secrets or API keys.
"""

from __future__ import annotations

import os
from typing import Optional
from snapshield.core.exceptions import PolicyViolationError, SecurityBoundaryError
from snapshield.core.logger import logger
from snapshield.core.types import PolicyAction, PolicyDecision, RedactionResult


class ExternalLLMGateway:
    """Security-guarded gateway for optional external AI forwarding.

    Acts as an egress perimeter firewall: guarantees that content cannot exit the device
    unless it has been evaluated and sanitized by the policy engine.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("EXTERNAL_LLM_API_KEY")
        self.model = model

    def forward_safe_prompt(
        self,
        sanitized_input: RedactionResult,
        policy_decision: PolicyDecision,
        system_instruction: str = "You are a helpful AI assistant.",
    ) -> str:
        """Transmits sanitized output to the configured external LLM service.

        Raises:
            PolicyViolationError: If policy action is BLOCK or REVIEW.
            SecurityBoundaryError: If the input is unverified or empty.
        """
        # Strict policy boundary check
        if policy_decision.action == PolicyAction.BLOCK:
            raise PolicyViolationError(
                "Egress blocked: Cannot forward content with active PolicyAction.BLOCK."
            )

        if policy_decision.action == PolicyAction.REVIEW:
            raise PolicyViolationError(
                "Egress paused: Content requires manual security review before external transmission."
            )

        if not sanitized_input or not sanitized_input.sanitized_text:
            raise SecurityBoundaryError("Cannot forward empty or unvalidated sanitized payload.")

        logger.info("Content passed security perimeter check (Hash: %s). Preparing for LLM egress.",
                    sanitized_input.integrity_hash[:8])

        # Note: In offline/stub mode or when no API key is set, returns simulated gateway response
        if not self.api_key:
            return (
                f"[SnapShield Gateway Demo Response] External LLM successfully received "
                f"sanitized prompt ({len(sanitized_input.sanitized_text)} chars, "
                f"{sanitized_input.redactions_applied} redactions applied). "
                f"Zero raw PII was transmitted."
            )

        # Actual downstream client call (e.g. OpenAI SDK) would execute here with sanitized_input.sanitized_text
        return f"[Live LLM Response Placeholder] Processed sanitized query: {sanitized_input.sanitized_text[:60]}..."
