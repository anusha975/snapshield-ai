"""SnapShield AI - External LLM Client with Enforced Safety Boundary.

Architecture:
ORIGINAL -> SCAN -> POLICY -> SANITIZE -> SAFETY CHECK -> LLM
NEVER: ORIGINAL -> LLM

Mandatory Pre-Flight Safety Checks before dispatching to an external LLM:
1. scan exists: A valid ScanResult must be supplied.
2. decisions exist: Policy decisions must be present and evaluated.
3. blocked entities are absent: Zero blocked entities or BLOCK policy actions permitted.
4. redaction is complete: Redaction has taken place and no sensitive tokens leak into sanitized text.
5. review items are approved: Any items requiring review have received explicit user approval.
6. sanitized text is used: The payload sent to the LLM must strictly be the sanitized text, NEVER original text.

Configuration:
- LLM_API_KEY: Authenticates external LLM calls. No hardcoded keys allowed.
- LLM_PROVIDER: Target provider (openai, anthropic, azure, groq, custom).
- LLM_MODEL: Model identifier (e.g. gpt-4o-mini).

Graceful Degradation:
If no LLM_API_KEY is configured in the environment, the feature is cleanly disabled
without raising crashes or halting the local privacy pipeline.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from snapshield.core.exceptions import (
    PolicyViolationError,
    SecurityBoundaryError,
    SnapShieldError,
)
from snapshield.core.logger import logger
from snapshield.core.models import PolicyAction, PolicyDecision, ScanResult
from snapshield.core.redactor import verify_sanitization_leak_free


class LLMClientError(SnapShieldError):
    """Raised when external LLM communication or response processing fails."""
    pass


@dataclass
class LLMConfig:
    """Strongly typed configuration for external LLM client loaded from environment."""
    api_key: Optional[str] = None
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Loads configuration strictly from environment variables without hardcoded secrets."""
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            api_key = api_key.strip()
            if not api_key:
                api_key = None

        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        base_url = os.getenv("EXTERNAL_LLM_BASE_URL") or os.getenv("LLM_BASE_URL")
        if base_url:
            base_url = base_url.strip()

        return cls(
            api_key=api_key,
            provider=provider,
            model=model,
            base_url=base_url,
        )

    @property
    def is_configured(self) -> bool:
        """Returns True if a valid non-empty API key is present."""
        return bool(self.api_key and len(self.api_key) > 0)

    @property
    def masked_api_key(self) -> Optional[str]:
        """Returns safe representation of the API key for diagnostics and UI (zero raw leaks)."""
        if not self.api_key:
            return None
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


@dataclass
class LLMResponse:
    """Strongly typed response outcome from external LLM processing."""
    content: str
    provider: str
    model: str
    success: bool
    is_external: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    sanitized_prompt_hash: str = ""
    status_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def safe_dict(self) -> Dict[str, Any]:
        """Safe dictionary representation for logs and telemetry."""
        return {
            "success": self.success,
            "provider": self.provider,
            "model": self.model,
            "is_external": self.is_external,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "sanitized_prompt_hash": self.sanitized_prompt_hash[:8] if self.sanitized_prompt_hash else "",
        }


def verify_safety_boundary(
    scan: Optional[ScanResult],
    text_to_send: str,
    review_approved: bool = False,
) -> None:
    """Enforces the mandatory 6-step pre-flight safety boundary before external LLM dispatch.

    Architecture:
        ORIGINAL -> SCAN -> POLICY -> SANITIZE -> SAFETY CHECK -> LLM
        NEVER: ORIGINAL -> LLM

    Pre-flight Checks:
        1. scan exists
        2. decisions exist
        3. blocked entities are absent
        4. redaction is complete
        5. review items are approved
        6. sanitized text is used

    Raises:
        SecurityBoundaryError: If any of the 6 safety checks fail.
        PolicyViolationError: If blocked content or unapproved review is attempted.
    """
    # -------------------------------------------------------------
    # Check 1: scan exists
    # -------------------------------------------------------------
    if scan is None:
        logger.critical("Security Boundary Violation: Attempted LLM dispatch with missing scan.")
        raise SecurityBoundaryError(
            "Safety Check 1/6 Failed: Scan does not exist. Content must undergo on-device "
            "scanning and policy evaluation before external transmission."
        )

    if not isinstance(scan, ScanResult):
        logger.critical("Security Boundary Violation: Invalid scan object type (%s).", type(scan))
        raise SecurityBoundaryError(
            f"Safety Check 1/6 Failed: Invalid scan object of type {type(scan).__name__}."
        )

    # -------------------------------------------------------------
    # Check 2: decisions exist
    # -------------------------------------------------------------
    if not scan.decisions:
        logger.critical("Security Boundary Violation: Scan has no policy decisions.")
        raise SecurityBoundaryError(
            "Safety Check 2/6 Failed: Policy decisions do not exist. Every detected entity "
            "and content payload must have evaluated policy decisions."
        )

    # -------------------------------------------------------------
    # Check 3: blocked entities are absent
    # -------------------------------------------------------------
    for decision in scan.decisions:
        if decision.action == PolicyAction.BLOCK:
            logger.warning(
                "Security Boundary Violation: Decision for entity '%s' is BLOCK.",
                getattr(decision, "entity_type", "UNKNOWN"),
            )
            raise PolicyViolationError(
                f"Safety Check 3/6 Failed: Entity category '{decision.entity_type}' is marked BLOCK. "
                "Outright egress prohibited."
            )

    if getattr(scan, "is_blocked", False) or scan.overall_action == PolicyAction.BLOCK:
        logger.warning("Security Boundary Violation: Blocked entities are present in scan.")
        raise PolicyViolationError(
            "Safety Check 3/6 Failed: Blocked entities are present in content. "
            "PolicyAction.BLOCK strictly prohibits external transmission."
        )

    # -------------------------------------------------------------
    # Check 4: redaction is complete
    # -------------------------------------------------------------
    if scan.sanitized_text is None:
        logger.critical("Security Boundary Violation: Sanitized text is None.")
        raise SecurityBoundaryError(
            "Safety Check 4/6 Failed: Redaction is incomplete. Sanitized text was not generated."
        )

    # Collect raw values of all REDACTED, BLOCKED, or REVIEW detections for leak verification
    # (Entities explicitly ALLOWED by the active policy profile are permitted in sanitized output)
    decisions_by_idx = {i: d.action for i, d in enumerate(scan.decisions)}
    sensitive_values_to_verify = set()
    for idx, det in enumerate(scan.detections):
        action = decisions_by_idx.get(idx, getattr(det, "action", PolicyAction.REDACT))
        if action != PolicyAction.ALLOW:
            raw = getattr(det, "raw_value", None)
            if raw and len(raw.strip()) >= 3:
                sensitive_values_to_verify.add(raw.strip())


    if sensitive_values_to_verify:
        try:
            verify_sanitization_leak_free(scan.sanitized_text, sensitive_values_to_verify)
        except Exception as exc:
            logger.critical("Security Boundary Violation: Incomplete redaction detected (%s).", exc)
            raise SecurityBoundaryError(
                f"Safety Check 4/6 Failed: Redaction is incomplete. Sensitive tokens persist in "
                f"sanitized text: {exc}"
            ) from exc

    # -------------------------------------------------------------
    # Check 5: review items are approved
    # -------------------------------------------------------------
    review_required = getattr(scan, "requires_review", False) or any(
        d.action == PolicyAction.REVIEW for d in scan.decisions
    )

    if review_required and not review_approved:
        logger.warning("Security Boundary Violation: Review items require explicit user approval.")
        raise PolicyViolationError(
            "Safety Check 5/6 Failed: Review items are present in content and require explicit "
            "user approval before external transmission."
        )

    # -------------------------------------------------------------
    # Check 6: sanitized text is used
    # -------------------------------------------------------------
    # Enforce rule: Never ORIGINAL -> LLM
    if hasattr(scan, "original_text") and scan.original_text:
        # If original text had detections that changed the text, text_to_send must NOT equal original_text
        if scan.detections and text_to_send == scan.original_text and scan.sanitized_text != scan.original_text:
            logger.critical("CRITICAL SAFETY BREACH: Attempted to transmit original unredacted text to external LLM!")
            raise SecurityBoundaryError(
                "Safety Check 6/6 Failed: CRITICAL SECURITY BREACH. Original unredacted text was "
                "provided for egress. Only decoupled sanitized text is permitted."
            )

    if text_to_send != scan.sanitized_text:
        logger.critical("Security Boundary Violation: Provided payload does not match approved sanitized text.")
        raise SecurityBoundaryError(
            "Safety Check 6/6 Failed: Payload mismatch. The text to send must strictly match "
            "the approved, decoupled sanitized text from the scan result."
        )

    # Additional double-check on text_to_send against sensitive values
    if sensitive_values_to_verify:
        content_outside_placeholders = re.sub(
            r"\[(?:REDACTED|BLOCKED|REVIEW_REQUIRED):\s*[^\]]+\]", "", text_to_send
        )
        for sensitive_val in sensitive_values_to_verify:
            if sensitive_val in content_outside_placeholders:
                logger.critical("Security Boundary Violation: Sensitive token found directly in transmission payload.")
                raise SecurityBoundaryError(
                    "Safety Check 6/6 Failed: Sanitized text payload contains sensitive tokens."
                )

    logger.info("Safety boundary verification passed: All 6 pre-flight egress checks verified.")


class ExternalLLMClient:
    """External LLM client enforcing strict security boundary and on-device privacy."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        # Load from arguments or fallback strictly to environment variables
        env_config = LLMConfig.from_env()
        self.config = LLMConfig(
            api_key=api_key if api_key is not None else env_config.api_key,
            provider=(provider or env_config.provider).lower(),
            model=model or env_config.model,
            base_url=base_url or env_config.base_url,
            timeout_seconds=timeout_seconds,
        )

    @property
    def is_configured(self) -> bool:
        """Returns True if LLM_API_KEY is configured in the environment or client."""
        return self.config.is_configured

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def masked_api_key(self) -> Optional[str]:
        return self.config.masked_api_key

    def get_status(self) -> Dict[str, Any]:
        """Provides transparent status for UI components without exposing credentials."""
        return {
            "configured": self.is_configured,
            "provider": self.config.provider,
            "model": self.config.model,
            "masked_api_key": self.config.masked_api_key,
            "is_external": True,
            "notice": "External cloud processing. Content transmitted leaves local device.",
        }

    def send_prompt(
        self,
        scan_result: ScanResult,
        text_to_send: str,
        review_approved: bool = False,
        system_prompt: str = "You are a helpful and concise AI assistant.",
    ) -> LLMResponse:
        """Transmits sanitized text to the external LLM provider after passing safety boundary.

        Enforces:
            ORIGINAL -> SCAN -> POLICY -> SANITIZE -> SAFETY CHECK -> LLM

        Args:
            scan_result: The completed on-device ScanResult.
            text_to_send: Must strictly be scan_result.sanitized_text.
            review_approved: Explicit approval if review items are present.
            system_prompt: System instruction prompt.

        Returns:
            LLMResponse containing provider output or graceful disabled notice.

        Raises:
            SecurityBoundaryError: If safety checks fail.
            PolicyViolationError: If blocked or unapproved content is attempted.
        """
        # 1. MANDATORY SAFETY BOUNDARY CHECK (6/6 checks)
        verify_safety_boundary(
            scan=scan_result,
            text_to_send=text_to_send,
            review_approved=review_approved,
        )

        sanitized_hash = getattr(scan_result, "integrity_hash", "")

        # 2. GRACEFUL DEGRADATION: If no API key configured, disable feature cleanly
        if not self.is_configured:
            logger.info("External LLM dispatch requested but LLM_API_KEY is not configured.")
            return LLMResponse(
                content="",
                provider=self.config.provider,
                model=self.config.model,
                success=False,
                is_external=True,
                error="External LLM feature is disabled: LLM_API_KEY is not configured in the environment.",
                sanitized_prompt_hash=sanitized_hash,
            )

        # 3. DISPATCH TO EXTERNAL PROVIDER
        start_time = time.perf_counter()
        try:
            response_text, status_code, metadata = self._dispatch_http_request(
                sanitized_prompt=text_to_send,
                system_prompt=system_prompt,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return LLMResponse(
                content=response_text,
                provider=self.config.provider,
                model=self.config.model,
                success=True,
                is_external=True,
                latency_ms=elapsed_ms,
                sanitized_prompt_hash=sanitized_hash,
                status_code=status_code,
                metadata=metadata,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("External LLM dispatch to %s failed: %s", self.config.provider, exc)
            return LLMResponse(
                content="",
                provider=self.config.provider,
                model=self.config.model,
                success=False,
                is_external=True,
                error=f"External LLM request failed: {str(exc)}",
                latency_ms=elapsed_ms,
                sanitized_prompt_hash=sanitized_hash,
            )

    def forward_sanitized(
        self,
        scan_result: ScanResult,
        review_approved: bool = False,
        system_prompt: str = "You are a helpful and concise AI assistant.",
    ) -> LLMResponse:
        """Convenience method sending scan_result.sanitized_text through safety boundary."""
        if scan_result is None:
            raise SecurityBoundaryError("Cannot forward None scan_result.")
        return self.send_prompt(
            scan_result=scan_result,
            text_to_send=scan_result.sanitized_text,
            review_approved=review_approved,
            system_prompt=system_prompt,
        )

    def _dispatch_http_request(
        self,
        sanitized_prompt: str,
        system_prompt: str,
    ) -> tuple[str, int, Dict[str, Any]]:
        """Dispatches HTTP request according to configured provider using urllib."""
        provider = self.config.provider
        api_key = self.config.api_key or ""
        model = self.config.model

        if provider in ("openai", "groq", "custom"):
            endpoint = self.config.base_url or (
                "https://api.groq.com/openai/v1/chat/completions"
                if provider == "groq"
                else "https://api.openai.com/v1/chat/completions"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "SnapShield-AI-Checkpoint/1.0",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sanitized_prompt},
                ],
                "temperature": 0.2,
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    status_code = resp.status
                    body = json.loads(resp.read().decode("utf-8"))
                    choices = body.get("choices", [])
                    content = choices[0].get("message", {}).get("content", "") if choices else ""
                    metadata = {"id": body.get("id"), "usage": body.get("usage", {})}
                    return content, status_code, metadata
            except urllib.error.HTTPError as http_err:
                if http_err.code in (401, 403):
                    from snapshield.core.exceptions import AuthenticationError
                    raise AuthenticationError(
                        f"Authentication failed with {provider} (HTTP {http_err.code}). "
                        f"Please check that your LLM_API_KEY is valid.",
                        details={"provider": provider, "status_code": http_err.code}
                    ) from http_err
                from snapshield.core.exceptions import LLMServiceError
                raise LLMServiceError(
                    f"External LLM service error from {provider} (HTTP {http_err.code}): {http_err.reason}",
                    details={"provider": provider, "status_code": http_err.code}
                ) from http_err
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as net_err:
                from snapshield.core.exceptions import NetworkError
                raise NetworkError(
                    f"Network connection to {provider} failed or timed out: {net_err}",
                    details={"provider": provider, "endpoint": endpoint}
                ) from net_err

        elif provider == "anthropic":
            endpoint = self.config.base_url or "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "SnapShield-AI-Checkpoint/1.0",
            }
            payload = {
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": sanitized_prompt}],
                "max_tokens": 1024,
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    status_code = resp.status
                    body = json.loads(resp.read().decode("utf-8"))
                    content_blocks = body.get("content", [])
                    content = content_blocks[0].get("text", "") if content_blocks else ""
                    metadata = {"id": body.get("id"), "usage": body.get("usage", {})}
                    return content, status_code, metadata
            except urllib.error.HTTPError as http_err:
                if http_err.code in (401, 403):
                    from snapshield.core.exceptions import AuthenticationError
                    raise AuthenticationError(
                        f"Authentication failed with Anthropic (HTTP {http_err.code}). "
                        f"Please verify your LLM_API_KEY.",
                        details={"provider": "anthropic", "status_code": http_err.code}
                    ) from http_err
                from snapshield.core.exceptions import LLMServiceError
                raise LLMServiceError(
                    f"Anthropic service error (HTTP {http_err.code}): {http_err.reason}",
                    details={"provider": "anthropic", "status_code": http_err.code}
                ) from http_err
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as net_err:
                from snapshield.core.exceptions import NetworkError
                raise NetworkError(
                    f"Network connection to Anthropic failed or timed out: {net_err}",
                    details={"provider": "anthropic", "endpoint": endpoint}
                ) from net_err

        else:
            # Generic HTTP / Mockable endpoint
            endpoint = self.config.base_url or f"https://api.{provider}.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sanitized_prompt},
                ],
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                status_code = resp.status
                body = json.loads(resp.read().decode("utf-8"))
                content = str(body)
                return content, status_code, {}


# Alias for clean importing
LLMClient = ExternalLLMClient


def get_llm_client() -> ExternalLLMClient:
    """Factory creating an ExternalLLMClient initialized from environment."""
    return ExternalLLMClient()
