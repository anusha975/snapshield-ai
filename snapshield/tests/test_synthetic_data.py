"""Tests validating synthetic test data and privacy rules.

Validates:
- Rule 7: Use synthetic data for testing.
- Luhn checksum calculation on synthetic payment cards.
- Separate original vs sanitized output.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snapshield.data import load_synthetic_samples
from snapshield.services.detector import HybridDetectionEngine, luhn_checksum_validate
from snapshield.services.policy import PolicyEngine
from snapshield.services.redactor import RedactionEngine
from snapshield.services.risk import RiskAssessmentEngine


def test_synthetic_data_loading():
    """Verify that synthetic data loads cleanly and contains required test scenarios."""
    samples = load_synthetic_samples()
    assert len(samples) >= 4
    for s in samples:
        assert "id" in s
        assert "name" in s
        assert "content" in s
        assert "expected_categories" in s


def test_luhn_algorithm_validation():
    """Verify Luhn checksum accurately passes valid cards and rejects invalid ones."""
    # Standard valid test card
    assert luhn_checksum_validate("4532015012345671") is True
    assert luhn_checksum_validate("4532-0150-1234-5671") is True
    # Invalid card number (tampered last digit)
    assert luhn_checksum_validate("4532015012345672") is False


def test_end_to_end_synthetic_checkpoint_flow():
    """Verifies full detection -> risk -> policy -> redaction pipeline on synthetic data."""
    samples = load_synthetic_samples()
    finance_sample = next(s for s in samples if s["id"] == "sample_finance_01")

    detector = HybridDetectionEngine()
    findings = detector.scan_text(finance_sample["content"])
    assert len(findings) > 0

    risk_engine = RiskAssessmentEngine()
    assessment = risk_engine.evaluate(findings)
    assert assessment.total_score > 0.0

    policy_engine = PolicyEngine(profile_name="STRICT")
    decision = policy_engine.decide(assessment)
    assert decision.action.value in ["REDACT", "BLOCK", "REVIEW"]

    # Redaction test: original string must NOT equal sanitized string
    redactor = RedactionEngine()
    redaction_res = redactor.redact_text(finance_sample["content"], findings)

    assert redaction_res.sanitized_text != finance_sample["content"]
    assert "[REDACTED_PAYMENT_CARD]" in redaction_res.sanitized_text
    assert redaction_res.integrity_hash != ""
