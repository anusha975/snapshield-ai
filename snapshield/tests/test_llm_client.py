"""Unit and integration tests for External LLM Client and Mandatory Safety Boundary.

Validates:
1. Mandatory Safety Boundary:
   - Check 1: scan exists
   - Check 2: decisions exist
   - Check 3: blocked entities are absent
   - Check 4: redaction is complete
   - Check 5: review items are approved
   - Check 6: sanitized text is used
   - Invariant: Never ORIGINAL -> LLM

2. Environment Configuration:
   - LLM_API_KEY, LLM_PROVIDER, LLM_MODEL loading
   - Zero hardcoded keys
   - Key masking in diagnostics
   - Graceful degradation when no key exists

3. Provider Dispatch:
   - OpenAI-compatible and Anthropic endpoints
   - Latency measurement
   - Clean error handling on HTTP failure
"""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.exceptions import (
    PolicyViolationError,
    SecurityBoundaryError,
)
from snapshield.core.llm_client import (
    ExternalLLMClient,
    LLMClient,
    LLMConfig,
    LLMResponse,
    get_llm_client,
    verify_safety_boundary,
)
from snapshield.core.models import (
    Detection,
    PolicyAction,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
    RuntimeStatus,
    ScanResult,
    ScanTiming,
    SeverityLevel,
)


def _build_dummy_runtime_status() -> RuntimeStatus:
    return RuntimeStatus(
        os_name="Windows",
        machine_arch="AMD64",
        processor="Intel x86_64",
        active_execution_provider="CPUExecutionProvider",
        is_accelerated=False,
        device_backend="Verified CPU Fallback",
    )


def _build_test_scan(
    original_text: str = "Patient Jane Doe, Aadhaar 9999 9999 0019, key sk-sec-1234567890abcdef",
    sanitized_text: str = "Patient Jane Doe, Aadhaar [REDACTED: AADHAAR], key [REDACTED: API_KEY]",
    action: PolicyAction = PolicyAction.REDACT,
    decisions: list[PolicyDecision] | None = None,
    detections: list[Detection] | None = None,
    requires_review: bool = False,
    is_blocked: bool = False,
) -> ScanResult:
    if detections is None:
        detections = [
            Detection(
                entity_type="AADHAAR",
                start=18,
                end=32,
                confidence=1.0,
                detector="regex",
                valid=True,
                severity=SeverityLevel.HIGH,
                raw_value="9999 9999 0019",
                explanation="Detected 12-digit Aadhaar number with valid Verhoeff checksum.",
            ),
            Detection(
                entity_type="API_KEY",
                start=38,
                end=68,
                confidence=1.0,
                detector="secret",
                valid=True,
                severity=SeverityLevel.CRITICAL,
                raw_value="sk-sec-1234567890abcdef",
                explanation="High-entropy secret key pattern matching provider credential token.",
            ),
        ]

    if decisions is None:
        decisions = [
            PolicyDecision(entity_type="AADHAAR", action=action, reason="Redact national ID"),
            PolicyDecision(entity_type="API_KEY", action=action, reason="Redact credential"),
        ]

    return ScanResult(
        original_text=original_text,
        sanitized_text=sanitized_text,
        detections=detections,
        decisions=decisions,
        risk_assessment=RiskAssessment(
            score=75.0,
            level=RiskLevel.HIGH,
            findings_count=len(detections),
            is_blocked=is_blocked,
        ),
        timings=ScanTiming(total_ms=12.5),
        runtime_status=_build_dummy_runtime_status(),
        integrity_hash="abc123sha256hash",
    )


# =========================================================================
# 1. Mandatory Safety Boundary Tests
# =========================================================================

def test_safety_boundary_check1_scan_must_exist():
    """Safety Check 1/6: Attempting to verify or send without a scan fails immediately."""
    with pytest.raises(SecurityBoundaryError, match="Scan does not exist"):
        verify_safety_boundary(scan=None, text_to_send="sanitized content")

    with pytest.raises(SecurityBoundaryError, match="Invalid scan object"):
        verify_safety_boundary(scan="not_a_scan_result", text_to_send="sanitized content")


def test_safety_boundary_check2_decisions_must_exist():
    """Safety Check 2/6: Scan without evaluated policy decisions fails."""
    scan = _build_test_scan(decisions=[])
    with pytest.raises(SecurityBoundaryError, match="Policy decisions do not exist"):
        verify_safety_boundary(scan=scan, text_to_send=scan.sanitized_text)


def test_safety_boundary_check3_blocked_entities_prohibited():
    """Safety Check 3/6: Blocked entities strictly prohibit external transmission."""
    # Test overall blocked
    blocked_scan = _build_test_scan(
        action=PolicyAction.BLOCK,
        decisions=[PolicyDecision(entity_type="API_KEY", action=PolicyAction.BLOCK, reason="Blocked key")],
        is_blocked=True,
    )
    with pytest.raises(PolicyViolationError, match="marked BLOCK|Blocked entities are present"):
        verify_safety_boundary(scan=blocked_scan, text_to_send=blocked_scan.sanitized_text)

    # Test individual decision blocked
    mixed_scan = _build_test_scan(
        decisions=[
            PolicyDecision(entity_type="AADHAAR", action=PolicyAction.REDACT, reason="Redact"),
            PolicyDecision(entity_type="PASSWORD", action=PolicyAction.BLOCK, reason="Block credentials"),
        ]
    )
    with pytest.raises(PolicyViolationError, match="marked BLOCK|Blocked entities are present"):
        verify_safety_boundary(scan=mixed_scan, text_to_send=mixed_scan.sanitized_text)


def test_safety_boundary_check4_redaction_completeness_and_leak_check():
    """Safety Check 4/6: Incomplete redaction or leaking sensitive tokens fails."""
    # Case A: sanitized_text is None
    leaky_scan = _build_test_scan()
    # Bypass pydantic validation post-init to verify defensive safety check
    object.__setattr__(leaky_scan, "sanitized_text", None)
    with pytest.raises(SecurityBoundaryError, match="Redaction is incomplete"):
        verify_safety_boundary(scan=leaky_scan, text_to_send="")

    # Case B: sanitized_text accidentally leaked raw secret outside placeholders
    accidental_leak_scan = _build_test_scan(
        sanitized_text="Aadhaar 9999 9999 0019 remained here accidentally sk-sec-1234567890abcdef",
    )
    with pytest.raises(SecurityBoundaryError, match="Sensitive tokens persist"):
        verify_safety_boundary(scan=accidental_leak_scan, text_to_send=accidental_leak_scan.sanitized_text)


def test_safety_boundary_check5_review_items_require_approval():
    """Safety Check 5/6: Items requiring review fail unless explicitly approved."""
    review_scan = _build_test_scan(
        action=PolicyAction.REVIEW,
        decisions=[PolicyDecision(entity_type="CONFIDENTIAL_INFO", action=PolicyAction.REVIEW, reason="Review required")],
    )

    # Unapproved fails
    with pytest.raises(PolicyViolationError, match="Review items are present"):
        verify_safety_boundary(scan=review_scan, text_to_send=review_scan.sanitized_text, review_approved=False)

    # Explicitly approved succeeds
    verify_safety_boundary(scan=review_scan, text_to_send=review_scan.sanitized_text, review_approved=True)


def test_safety_boundary_check6_sanitized_text_is_strictly_used():
    """Safety Check 6/6: Attempting to send original text instead of sanitized text is rejected."""
    scan = _build_test_scan()

    # Attempting to send original text
    with pytest.raises(SecurityBoundaryError, match="CRITICAL SECURITY BREACH"):
        verify_safety_boundary(scan=scan, text_to_send=scan.original_text)

    # Attempting to send modified arbitrary payload that doesn't match sanitized text
    with pytest.raises(SecurityBoundaryError, match="Payload mismatch"):
        verify_safety_boundary(scan=scan, text_to_send="completely different text")

    # Correct sanitized text passes smoothly
    verify_safety_boundary(scan=scan, text_to_send=scan.sanitized_text)


# =========================================================================
# 2. Environment Configuration & Graceful Degradation
# =========================================================================

def test_environment_configuration_loading(monkeypatch):
    """Verifies that LLMConfig accurately reflects LLM_API_KEY, LLM_PROVIDER, LLM_MODEL."""
    monkeypatch.setenv("LLM_API_KEY", "sk-proj-test1234567890xyz")
    monkeypatch.setenv("LLM_PROVIDER", "Anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet")

    config = LLMConfig.from_env()
    assert config.is_configured is True
    assert config.provider == "anthropic"
    assert config.model == "claude-3-5-sonnet"
    assert config.masked_api_key == "sk-p...0xyz"


def test_graceful_degradation_when_no_api_key(monkeypatch):
    """When LLM_API_KEY is missing, feature is cleanly disabled without crashing."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    client = ExternalLLMClient()
    assert client.is_configured is False
    assert client.masked_api_key is None

    scan = _build_test_scan()
    response = client.forward_sanitized(scan)

    assert isinstance(response, LLMResponse)
    assert response.success is False
    assert "LLM_API_KEY is not configured" in response.error
    assert response.is_external is True


def test_client_status_does_not_leak_raw_api_key(monkeypatch):
    """get_status() returns transparent metadata with masked key."""
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key-123456")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    client = ExternalLLMClient()
    status = client.get_status()

    assert status["configured"] is True
    assert status["provider"] == "openai"
    assert status["model"] == "gpt-4o"
    assert "super-secret-key-123456" not in str(status)
    assert status["masked_api_key"] == "supe...3456"


# =========================================================================
# 3. Provider Dispatch & Mocked HTTP Execution
# =========================================================================

def test_successful_dispatch_openai_format(monkeypatch):
    """Verifies dispatch to OpenAI-compatible provider with timing and response parsing."""
    monkeypatch.setenv("LLM_API_KEY", "sk-valid-key-12345678")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    client = ExternalLLMClient()
    scan = _build_test_scan()

    mock_resp_data = {
        "id": "chatcmpl-test-123",
        "choices": [{"message": {"content": "Here is the safe clinical summary."}}],
        "usage": {"total_tokens": 42},
    }

    mock_urlopen = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value.status = 200
    mock_cm.__enter__.return_value.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
    mock_urlopen.return_value = mock_cm

    with patch("urllib.request.urlopen", mock_urlopen):
        resp = client.forward_sanitized(scan)

        assert resp.success is True
        assert resp.content == "Here is the safe clinical summary."
        assert resp.provider == "openai"
        assert resp.model == "gpt-4o-mini"
        assert resp.latency_ms > 0.0
        assert resp.status_code == 200

        # Verify outgoing request payload used sanitized text, not original text
        call_args = mock_urlopen.call_args[0]
        req_obj = call_args[0]
        sent_body = json.loads(req_obj.data.decode("utf-8"))

        user_message = sent_body["messages"][1]["content"]
        assert user_message == scan.sanitized_text
        assert "sk-sec-1234567890abcdef" not in user_message


def test_successful_dispatch_anthropic_format(monkeypatch):
    """Verifies dispatch to Anthropic provider with custom headers and payload."""
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-test-key-99887766")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-haiku")

    client = ExternalLLMClient()
    scan = _build_test_scan()

    mock_resp_data = {
        "id": "msg-test-123",
        "content": [{"text": "Anthropic processed the sanitized text safely."}],
    }

    mock_urlopen = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value.status = 200
    mock_cm.__enter__.return_value.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
    mock_urlopen.return_value = mock_cm

    with patch("urllib.request.urlopen", mock_urlopen):
        resp = client.forward_sanitized(scan)

        assert resp.success is True
        assert resp.content == "Anthropic processed the sanitized text safely."
        assert resp.provider == "anthropic"


def test_http_failure_handled_cleanly(monkeypatch):
    """Network or API error does not raise uncaught exception, returns informative LLMResponse."""
    monkeypatch.setenv("LLM_API_KEY", "sk-valid-key")
    client = ExternalLLMClient()
    scan = _build_test_scan()

    with patch("urllib.request.urlopen", side_effect=Exception("Connection timed out")):
        resp = client.forward_sanitized(scan)
        assert resp.success is False
        assert "Connection timed out" in resp.error
        assert resp.is_external is True
