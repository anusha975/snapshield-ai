"""Comprehensive unit tests for the Sanitized Privacy Report system.

Validates:
- Precise calculation of Summary metrics:
  * ENTITIES DETECTED
  * ENTITIES REDACTED
  * ENTITIES BLOCKED
  * ENTITIES ALLOWED
  * REVIEW REQUIRED
- Risk score and qualitative tier reporting.
- Detection method breakdown: Regex, Checksum, OCR, AI.
- Performance latency breakdown: OCR, Detection, AI, Total.
- Runtime hardware and model diagnostics.
- External processing status: LLM request: Yes/No.
- Strict invariant: Zero raw sensitive values in report or exported JSON.
- Export of valid, sanitized JSON reports.
- Streamlit UI rendering component stability.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

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
from snapshield.core.privacy_report import (
    SanitizedPrivacyReport,
    generate_sanitized_privacy_report,
)
from snapshield.ui.components import render_privacy_report


@pytest.fixture
def sample_scan_result() -> ScanResult:
    """Creates a realistic, multi-entity ScanResult for privacy report verification."""
    detections = [
        Detection(
            entity_type="AADHAAR",
            start=10,
            end=24,
            confidence=1.0,
            detector="verhoeff",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Validated Aadhaar number via Verhoeff checksum.",
            raw_value="9999 9999 0019",
        ),
        Detection(
            entity_type="OPENAI_KEY",
            start=40,
            end=91,
            confidence=0.99,
            detector="regex",
            valid=True,
            severity=SeverityLevel.CRITICAL,
            explanation="OpenAI API project key detected.",
            raw_value="sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678",
        ),
        Detection(
            entity_type="MEDICAL_INFORMATION",
            start=110,
            end=135,
            confidence=0.94,
            detector="local_ai_onnx",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Contextual diagnosis detected via on-device model.",
            raw_value="acute coronary syndrome",
        ),
        Detection(
            entity_type="PUBLIC_ORGANIZATION",
            start=150,
            end=170,
            confidence=0.88,
            detector="regex",
            valid=True,
            severity=SeverityLevel.LOW,
            explanation="Public institutional name permitted.",
            raw_value="City General Hospital",
        ),
    ]

    decisions = [
        PolicyDecision(entity_type="AADHAAR", action=PolicyAction.REDACT, reason="Mask national identity numbers."),
        PolicyDecision(entity_type="OPENAI_KEY", action=PolicyAction.BLOCK, reason="Outright prohibit API keys from egress."),
        PolicyDecision(entity_type="MEDICAL_INFORMATION", action=PolicyAction.REDACT, reason="Scrub clinical symptoms."),
        PolicyDecision(entity_type="PUBLIC_ORGANIZATION", action=PolicyAction.ALLOW, reason="Public entity permitted."),
    ]

    return ScanResult(
        original_text="Patient: 9999 9999 0019, key sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678, diag: acute coronary syndrome, org: City General Hospital",
        sanitized_text="Patient: [REDACTED: AADHAAR], key [REDACTED: OPENAI_KEY], diag: [REDACTED: MEDICAL], org: City General Hospital",
        detections=detections,
        decisions=decisions,
        risk_assessment=RiskAssessment(
            score=82.5,
            level=RiskLevel.CRITICAL,
            findings_count=4,
            is_blocked=True,
            reasons=["Critical API key detected", "High-risk national identifier discovered"],
        ),
        timings=ScanTiming(
            extraction_ms=1.2,
            ocr_ms=45.6,
            structured_detection_ms=3.4,
            ai_inference_ms=12.8,
            risk_ms=0.5,
            policy_ms=0.4,
            redaction_ms=0.6,
            total_ms=64.5,
        ),
        runtime_status=RuntimeStatus(
            os_name="Windows",
            machine_arch="AMD64",
            processor="Intel64",
            active_execution_provider="CPUExecutionProvider",
            is_accelerated=False,
            device_backend="CPUExecutionProvider",
        ),
    )


def test_generate_sanitized_privacy_report_summary(sample_scan_result):
    """Verifies that privacy report computes exact summary counts correctly."""
    report = generate_sanitized_privacy_report(
        scan_result=sample_scan_result,
        llm_requested=False,
    )

    assert isinstance(report, SanitizedPrivacyReport)

    # Summary Counters
    assert report.summary.entities_detected == 4
    assert report.summary.entities_redacted == 2  # AADHAAR, MEDICAL_INFORMATION
    assert report.summary.entities_blocked == 1   # OPENAI_KEY
    assert report.summary.entities_allowed == 1   # PUBLIC_ORGANIZATION
    assert report.summary.review_required == "No"

    # Risk
    assert report.risk.score == 82.5
    assert report.risk.level == "CRITICAL"
    assert report.risk.is_blocked is True
    assert len(report.risk.reasons) == 2


def test_generate_sanitized_privacy_report_detection_breakdown(sample_scan_result):
    """Verifies that detection extraction methods are categorized accurately."""
    report = generate_sanitized_privacy_report(
        scan_result=sample_scan_result,
        llm_requested=False,
    )

    # Breakdown:
    # AADHAAR -> Checksum
    # OPENAI_KEY -> Regex
    # MEDICAL_INFORMATION -> AI
    # PUBLIC_ORGANIZATION -> Regex
    assert report.detection_breakdown.checksum == 1
    assert report.detection_breakdown.ai == 1
    assert report.detection_breakdown.regex == 2
    assert report.detection_breakdown.ocr == 0


def test_generate_sanitized_privacy_report_performance(sample_scan_result):
    """Verifies performance latency metrics match pipeline measurements."""
    report = generate_sanitized_privacy_report(
        scan_result=sample_scan_result,
        llm_requested=False,
    )

    assert report.performance.ocr_ms == 45.6
    assert report.performance.detection_ms == 3.4
    assert report.performance.ai_inference_ms == 12.8
    assert report.performance.total_ms == 64.5


def test_generate_sanitized_privacy_report_runtime_and_external(sample_scan_result):
    """Verifies runtime diagnostics and external processing indicators."""
    report_local = generate_sanitized_privacy_report(
        scan_result=sample_scan_result,
        llm_requested=False,
    )

    assert report_local.runtime.model == "bert-small-pii-detection-int8"
    assert report_local.runtime.execution_provider == "CPUExecutionProvider"
    assert "Verified CPU Fallback" in report_local.runtime.local_processing_status
    assert report_local.external_processing.llm_request == "No"

    report_external = generate_sanitized_privacy_report(
        scan_result=sample_scan_result,
        llm_requested=True,
    )
    assert report_external.external_processing.llm_request == "Yes"


def test_privacy_report_never_includes_raw_sensitive_values(sample_scan_result):
    """Strict security invariant: JSON export must NEVER contain raw sensitive values."""
    report = generate_sanitized_privacy_report(
        scan_result=sample_scan_result,
        llm_requested=False,
    )

    json_output = report.to_json()

    # Raw sensitive tokens that must NEVER appear in JSON report
    raw_forbidden_tokens = [
        "9999 9999 0019",
        "sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678",
        "Patient: 9999 9999 0019",
    ]

    for token in raw_forbidden_tokens:
        assert token not in json_output, f"Raw secret leaked into exported JSON report: {token}"

    # Verify JSON is valid parseable JSON
    parsed = json.loads(json_output)
    assert parsed["report_id"] == sample_scan_result.scan_id
    assert parsed["summary"]["entities_detected"] == 4
    assert len(parsed["findings"]) == 4

    # Check that finding previews are properly masked
    for finding in parsed["findings"]:
        assert "raw_value" not in finding
        assert "masked_preview" in finding
        assert not any(t in finding["masked_preview"] for t in raw_forbidden_tokens)


def test_render_privacy_report_ui_component(sample_scan_result):
    """Verifies render_privacy_report renders all required visual sections and export button."""
    def mock_cols(spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(count)]

    with patch("streamlit.markdown") as mock_markdown, \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.columns", side_effect=mock_cols), \
         patch("streamlit.download_button") as mock_download_btn:

        render_privacy_report(sample_scan_result, llm_requested=False)

        assert mock_metric.called
        assert mock_markdown.called
        assert mock_download_btn.called

        # Verify download button parameters
        dl_kwargs = mock_download_btn.call_args[1]
        assert "snapshield_privacy_report_" in dl_kwargs["file_name"]
        assert dl_kwargs["mime"] == "application/json"
        assert dl_kwargs["label"] == "📥 Export Sanitized JSON Report"
