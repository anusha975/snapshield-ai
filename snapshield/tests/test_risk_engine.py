"""Comprehensive tests for privacy risk evaluation engine (core/risk_engine.py).

Validates:
1. 0–100 composite risk score calculation and bounds.
2. 4 Risk level tiers:
   - LOW
   - MEDIUM
   - HIGH
   - CRITICAL
3. Configurable weights, decay factors, and thresholds without unexplained hardcoding.
4. Potential severity mapping:
   - Credentials -> CRITICAL
   - Government IDs -> HIGH
   - Financial -> HIGH
   - Medical -> HIGH
   - Phone -> MEDIUM
   - Email -> MEDIUM
   - Organization -> LOW
5. Exact test verification of specification example:
   Risk Score: 82
   HIGH
   Contributors:
   Government ID +30
   Financial +25
   Medical +20
   Phone +7
6. Strict privacy guarantee: Zero exposure of raw sensitive values.
7. Graceful handling of empty or legacy finding inputs.
"""

from __future__ import annotations

import pytest

from snapshield.core.models import Detection, RiskLevel, SeverityLevel
from snapshield.core.risk_engine import (
    DEFAULT_CATEGORY_WEIGHTS,
    DEFAULT_RISK_THRESHOLDS,
    RiskContributor,
    RiskEngine,
    RiskEngineConfig,
    RiskReport,
    calculate_risk,
)


# =====================================================================
# 1. Specification Example Verification (Risk Score: 82, HIGH)
# =====================================================================

def test_specification_example_risk_score_82_high():
    """Verify the exact example from user specification:

    Risk Score: 82
    HIGH

    Contributors:
    Government ID +30
    Financial +25
    Medical +20
    Phone +7
    """
    findings = [
        Detection(
            entity_type="Aadhaar",
            start=10,
            end=24,
            confidence=1.0,
            detector="regex_aadhaar",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Indian Aadhaar national identifier.",
            raw_value="999912345678",
        ),
        Detection(
            entity_type="Bank Account",
            start=40,
            end=55,
            confidence=1.0,
            detector="regex_bank_account",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Personal bank account number.",
            raw_value="987654321012",
        ),
        Detection(
            entity_type="MEDICAL_INFORMATION",
            start=70,
            end=95,
            confidence=1.0,
            detector="local_ai_onnx",
            valid=True,
            severity=SeverityLevel.HIGH,
            explanation="Clinical diagnostic finding.",
            raw_value="diagnosed with hypertension",
        ),
        Detection(
            entity_type="Phone",
            start=110,
            end=122,
            confidence=1.0,
            detector="regex_phone",
            valid=True,
            severity=SeverityLevel.MEDIUM,
            explanation="Contact phone number.",
            raw_value="+919876543210",
        ),
    ]

    report = calculate_risk(findings)

    # 1. Verify score is exactly 82
    assert report.score == 82.0

    # 2. Verify level is HIGH
    assert report.level == RiskLevel.HIGH
    assert report.level.value == "HIGH"

    # 3. Verify contributors
    assert len(report.contributors) == 4

    contributor_map = {c.category: c.points for c in report.contributors}
    assert contributor_map["Government ID"] == 30.0
    assert contributor_map["Financial"] == 25.0
    assert contributor_map["Medical"] == 20.0
    assert contributor_map["Phone"] == 7.0

    # 4. Verify formatted display strings
    display_list = report.formatted_contributors
    assert "Government ID +30" in display_list
    assert "Financial +25" in display_list
    assert "Medical +20" in display_list
    assert "Phone +7" in display_list

    # 5. Verify summary output text format
    summary_text = report.summary()
    assert "Risk Score: 82" in summary_text
    assert "HIGH" in summary_text
    assert "Government ID +30" in summary_text
    assert "Financial +25" in summary_text
    assert "Medical +20" in summary_text
    assert "Phone +7" in summary_text


# =====================================================================
# 2. Potential Severity Alignment Tests
# =====================================================================

def test_potential_severity_alignments():
    """Verify category severities align with specification:

    Credentials -> CRITICAL
    Government IDs -> HIGH
    Financial -> HIGH
    Medical -> HIGH
    Phone -> MEDIUM
    Email -> MEDIUM
    Organization -> LOW
    """
    engine = RiskEngine()

    test_cases = [
        ("API_KEY", "Credentials", SeverityLevel.CRITICAL),
        ("PASSWORD", "Credentials", SeverityLevel.CRITICAL),
        ("DATABASE_URI", "Credentials", SeverityLevel.CRITICAL),
        ("AADHAAR", "Government ID", SeverityLevel.HIGH),
        ("PAN", "Government ID", SeverityLevel.HIGH),
        ("BANK_ACCOUNT", "Financial", SeverityLevel.HIGH),
        ("PAYMENT_CARD", "Financial", SeverityLevel.HIGH),
        ("MEDICAL_INFORMATION", "Medical", SeverityLevel.HIGH),
        ("PATIENT_IDENTIFIER", "Medical", SeverityLevel.HIGH),
        ("PHONE", "Phone", SeverityLevel.MEDIUM),
        ("EMAIL", "Email", SeverityLevel.MEDIUM),
        ("ORGANIZATION", "Organization", SeverityLevel.LOW),
    ]

    for raw_type, expected_cat, expected_sev in test_cases:
        cat_name = engine.resolve_category(raw_type)
        assert cat_name == expected_cat
        assert engine.config.category_severity[cat_name] == expected_sev


# =====================================================================
# 3. Risk Level Tiers & Thresholds (LOW, MEDIUM, HIGH, CRITICAL)
# =====================================================================

def test_risk_level_tiers():
    """Verify tier assignments for LOW, MEDIUM, HIGH, and CRITICAL thresholds."""
    engine = RiskEngine()

    # Case 1: LOW (< 25)
    low_findings = [
        Detection(
            entity_type="Email",
            start=0,
            end=10,
            confidence=1.0,
            detector="test",
            explanation="Single email",
        )
    ]
    report_low = engine.evaluate(low_findings)
    assert report_low.score == 5.0
    assert report_low.level == RiskLevel.LOW

    # Case 2: MEDIUM (25 <= score < 60)
    med_findings = [
        Detection(
            entity_type="Address",
            start=0,
            end=10,
            confidence=1.0,
            detector="test",
            explanation="Address 1",
        ),
        Detection(
            entity_type="Person",
            start=20,
            end=30,
            confidence=1.0,
            detector="test",
            explanation="Person 1",
        ),
        Detection(
            entity_type="Phone",
            start=40,
            end=50,
            confidence=1.0,
            detector="test",
            explanation="Phone 1",
        ),
    ]  # 10 + 8 + 7 = 25.0
    report_med = engine.evaluate(med_findings)
    assert report_med.score == 25.0
    assert report_med.level == RiskLevel.MEDIUM

    # Case 3: HIGH (60 <= score < 85)
    high_findings = [
        Detection(
            entity_type="Government ID",
            start=0,
            end=10,
            confidence=1.0,
            detector="test",
            explanation="Gov ID",
        ),
        Detection(
            entity_type="Financial",
            start=20,
            end=30,
            confidence=1.0,
            detector="test",
            explanation="Financial",
        ),
        Detection(
            entity_type="Medical",
            start=40,
            end=50,
            confidence=1.0,
            detector="test",
            explanation="Medical",
        ),
    ]  # 30 + 25 + 20 = 75.0
    report_high = engine.evaluate(high_findings)
    assert report_high.score == 75.0
    assert report_high.level == RiskLevel.HIGH

    # Case 4: CRITICAL (score >= 85 or auto-escalated on leaked credentials)
    crit_findings = [
        Detection(
            entity_type="API_KEY",
            start=0,
            end=30,
            confidence=1.0,
            detector="test",
            severity=SeverityLevel.CRITICAL,
            explanation="Leaked API Key",
        ),
        Detection(
            entity_type="Government ID",
            start=40,
            end=50,
            confidence=1.0,
            detector="test",
            explanation="Gov ID",
        ),
        Detection(
            entity_type="Financial",
            start=60,
            end=70,
            confidence=1.0,
            detector="test",
            explanation="Bank",
        ),
    ]  # 40 + 30 + 25 = 95.0
    report_crit = engine.evaluate(crit_findings)
    assert report_crit.score == 95.0
    assert report_crit.level == RiskLevel.CRITICAL
    assert report_crit.is_blocked is True


# =====================================================================
# 4. Configurable Weights & Customization Tests
# =====================================================================

def test_configurable_weights_and_thresholds():
    """Verify that custom weights and thresholds alter risk computation as configured."""
    custom_weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    custom_weights["Phone"] = 25.0  # Elevate phone weight in custom profile

    custom_thresholds = dict(DEFAULT_RISK_THRESHOLDS)
    custom_thresholds["HIGH"] = 50.0  # Lower high threshold

    custom_config = RiskEngineConfig(
        category_weights=custom_weights,
        thresholds=custom_thresholds,
    )
    engine = RiskEngine(config=custom_config)

    findings = [
        Detection(
            entity_type="Phone",
            start=0,
            end=10,
            confidence=1.0,
            detector="test",
            explanation="Phone with custom weight",
        ),
        Detection(
            entity_type="Government ID",
            start=20,
            end=30,
            confidence=1.0,
            detector="test",
            explanation="Gov ID",
        ),
    ]  # 25 + 30 = 55.0

    report = engine.evaluate(findings)
    assert report.score == 55.0
    # With custom high threshold of 50.0, 55.0 triggers HIGH
    assert report.level == RiskLevel.HIGH
    contributor_map = {c.category: c.points for c in report.contributors}
    assert contributor_map["Phone"] == 25.0


def test_diminishing_returns_on_repeat_findings():
    """Verify that multiple findings of same category apply diminishing returns factor."""
    config = RiskEngineConfig(diminishing_returns_factor=0.5)
    engine = RiskEngine(config=config)

    # 2 Phone findings: 1st gives 7.0, 2nd gives 7.0 * 0.5 = 3.5 -> Total = 10.5
    findings = [
        Detection(
            entity_type="Phone",
            start=0,
            end=10,
            confidence=1.0,
            detector="test",
            explanation="Phone 1",
        ),
        Detection(
            entity_type="Phone",
            start=20,
            end=30,
            confidence=1.0,
            detector="test",
            explanation="Phone 2",
        ),
    ]

    report = engine.evaluate(findings)
    assert report.score == 10.5
    assert len(report.contributors) == 1
    assert report.contributors[0].category == "Phone"
    assert report.contributors[0].count == 2
    assert report.contributors[0].points == 10.5


# =====================================================================
# 5. Strict Privacy Guarantee: Zero Exposure of Raw Values
# =====================================================================

def test_never_expose_raw_values_in_report_or_contributors():
    """Verify raw secret values are NEVER exposed in score, report, contributors, or summaries."""
    secret_value = "sk-proj-superSecretLiveProductionApiKey9999"
    findings = [
        Detection(
            entity_type="API_KEY",
            start=0,
            end=40,
            confidence=1.0,
            detector="regex_openai",
            valid=True,
            severity=SeverityLevel.CRITICAL,
            explanation="API Key finding.",
            raw_value=secret_value,
        )
    ]

    report = calculate_risk(findings)

    # 1. Check RiskReport string representation
    assert secret_value not in str(report)
    assert secret_value not in repr(report)

    # 2. Check summary text
    assert secret_value not in report.summary()

    # 3. Check each contributor
    for c in report.contributors:
        assert secret_value not in str(c)
        assert secret_value not in repr(c)
        assert secret_value not in c.display()
        assert secret_value not in c.explanation


# =====================================================================
# 6. Edge Cases & Robustness
# =====================================================================

def test_empty_findings_list():
    """Verify empty detection list produces 0 score and LOW risk."""
    report = calculate_risk([])
    assert report.score == 0.0
    assert report.level == RiskLevel.LOW
    assert report.contributors == []
    assert report.findings_count == 0
    assert report.is_blocked is False


def test_score_never_exceeds_100():
    """Verify that even with an overwhelming number of high-severity findings, score is capped at 100.0."""
    findings = [
        Detection(
            entity_type="API_KEY",
            start=i * 20,
            end=(i + 1) * 20,
            confidence=1.0,
            detector="test",
            severity=SeverityLevel.CRITICAL,
            explanation="Key",
        )
        for i in range(15)
    ]

    report = calculate_risk(findings)
    assert report.score == 50.0  # Max category cap applied (50.0)

    # Now add multiple other categories
    findings.extend(
        [
            Detection(
                entity_type="Aadhaar",
                start=500,
                end=512,
                confidence=1.0,
                detector="test",
                severity=SeverityLevel.HIGH,
                explanation="Aadhaar",
            ),
            Detection(
                entity_type="Bank Account",
                start=520,
                end=535,
                confidence=1.0,
                detector="test",
                severity=SeverityLevel.HIGH,
                explanation="Bank",
            ),
            Detection(
                entity_type="Medical",
                start=550,
                end=570,
                confidence=1.0,
                detector="test",
                severity=SeverityLevel.HIGH,
                explanation="Medical",
            ),
        ]
    )

    full_report = calculate_risk(findings)
    assert full_report.score <= 100.0
    assert full_report.score >= 85.0
    assert full_report.level == RiskLevel.CRITICAL


def test_to_risk_assessment_conversion():
    """Verify conversion to domain model RiskAssessment for pipeline compatibility."""
    findings = [
        Detection(
            entity_type="Phone",
            start=0,
            end=10,
            confidence=1.0,
            detector="test",
            explanation="Phone",
        )
    ]
    report = calculate_risk(findings)
    assessment = report.to_risk_assessment()

    assert assessment.score == 7.0
    assert assessment.level == RiskLevel.LOW
    assert assessment.findings_count == 1
    assert "Phone" in assessment.category_breakdown
