"""Comprehensive tests for the policy enforcement subsystem (core/policy.py).

Validates:
1. Supported policy modes:
   - Personal
   - Developer
   - Medical
2. Policy actions:
   - ALLOW
   - REDACT
   - BLOCK
   - REVIEW
3. Central policy loading from config/policies.json
4. apply_policy(detections, mode) contract:
   - Every detection receives a decision (1:1 mapping)
   - Every decision contains a human-readable reason
   - Invariant: Unknown sensitive categories strictly default to REVIEW
5. Precedence rules for document-level aggregate decisions (BLOCK > REVIEW > REDACT > ALLOW)
6. Case and formatting normalization across category names
"""

from __future__ import annotations

import pytest

from snapshield.core.models import Detection, PolicyAction, PolicyDecision, SeverityLevel
from snapshield.core.policy import (
    SUPPORTED_MODES,
    PolicyEngine,
    apply_policy,
)


# =====================================================================
# 1. Personal Mode Policy Evaluation
# =====================================================================

def test_personal_mode_policy_decisions():
    """Verify Personal mode policy rules:

    Aadhaar → REDACT
    PAN → REDACT
    Phone → REDACT
    Bank Account → REDACT
    IFSC → REDACT
    Email → ALLOW
    Medical Information → REDACT
    """
    findings = [
        Detection(entity_type="Aadhaar", start=0, end=12, confidence=1.0, detector="regex", explanation="Aadhaar"),
        Detection(entity_type="PAN", start=15, end=25, confidence=1.0, detector="regex", explanation="PAN"),
        Detection(entity_type="Phone", start=30, end=40, confidence=1.0, detector="regex", explanation="Phone"),
        Detection(entity_type="Bank Account", start=45, end=60, confidence=1.0, detector="regex", explanation="Bank"),
        Detection(entity_type="IFSC", start=65, end=76, confidence=1.0, detector="regex", explanation="IFSC"),
        Detection(entity_type="Email", start=80, end=95, confidence=1.0, detector="regex", explanation="Email"),
        Detection(entity_type="Medical Information", start=100, end=120, confidence=1.0, detector="ai", explanation="Medical"),
    ]

    decisions = apply_policy(findings, mode="Personal")

    # Every detection must receive a decision
    assert len(decisions) == len(findings)

    expected_actions = [
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.ALLOW,
        PolicyAction.REDACT,
    ]

    for d, exp_action, finding in zip(decisions, expected_actions, findings):
        assert d.entity_type == finding.entity_type
        assert d.action == exp_action
        assert d.profile_applied == "Personal"
        # Every decision must contain a human-readable reason
        assert len(d.reason) > 10
        assert "Personal" in d.reason


# =====================================================================
# 2. Developer Mode Policy Evaluation
# =====================================================================

def test_developer_mode_policy_decisions():
    """Verify Developer mode policy rules:

    API_KEY → BLOCK
    PASSWORD → BLOCK
    JWT → BLOCK
    Email → ALLOW
    IP_ADDRESS → REVIEW
    Code → ALLOW
    """
    findings = [
        Detection(entity_type="API_KEY", start=0, end=30, confidence=1.0, detector="cred", explanation="OpenAI key"),
        Detection(entity_type="PASSWORD", start=35, end=50, confidence=1.0, detector="cred", explanation="Root password"),
        Detection(entity_type="JWT", start=55, end=100, confidence=1.0, detector="cred", explanation="JWT auth token"),
        Detection(entity_type="Email", start=105, end=120, confidence=1.0, detector="regex", explanation="Dev email"),
        Detection(entity_type="IP_ADDRESS", start=125, end=138, confidence=1.0, detector="regex", explanation="Internal IP"),
        Detection(entity_type="Code", start=140, end=180, confidence=1.0, detector="regex", explanation="Snippet"),
    ]

    decisions = apply_policy(findings, mode="Developer")
    assert len(decisions) == len(findings)

    expected_actions = [
        PolicyAction.BLOCK,
        PolicyAction.BLOCK,
        PolicyAction.BLOCK,
        PolicyAction.ALLOW,
        PolicyAction.REVIEW,
        PolicyAction.ALLOW,
    ]

    for d, exp_action, finding in zip(decisions, expected_actions, findings):
        assert d.entity_type == finding.entity_type
        assert d.action == exp_action
        assert d.profile_applied == "Developer"
        assert len(d.reason) > 10
        assert "Developer" in d.reason


# =====================================================================
# 3. Medical Mode Policy Evaluation
# =====================================================================

def test_medical_mode_policy_decisions():
    """Verify optional Medical mode policy rules:

    Patient Identifier → BLOCK
    Medical Information → REDACT
    Aadhaar → REDACT
    Phone → REDACT
    Email → REDACT
    Address → REDACT
    """
    findings = [
        Detection(entity_type="Patient Identifier", start=0, end=15, confidence=1.0, detector="ai", explanation="MRN"),
        Detection(entity_type="Medical Information", start=20, end=40, confidence=1.0, detector="ai", explanation="Diagnosis"),
        Detection(entity_type="Aadhaar", start=45, end=57, confidence=1.0, detector="regex", explanation="Gov ID"),
        Detection(entity_type="Phone", start=60, end=70, confidence=1.0, detector="regex", explanation="Phone"),
        Detection(entity_type="Email", start=75, end=90, confidence=1.0, detector="regex", explanation="Email"),
        Detection(entity_type="Address", start=95, end=120, confidence=1.0, detector="ai", explanation="Home address"),
    ]

    decisions = apply_policy(findings, mode="Medical")
    assert len(decisions) == len(findings)

    expected_actions = [
        PolicyAction.BLOCK,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
        PolicyAction.REDACT,
    ]

    for d, exp_action in zip(decisions, expected_actions):
        assert d.action == exp_action
        assert d.profile_applied == "Medical"
        assert len(d.reason) > 10


# =====================================================================
# 4. Unknown Sensitive Categories Strictly Default to REVIEW
# =====================================================================

def test_unknown_categories_default_to_review_personal():
    """Verify that sensitive categories not in Personal profile default to REVIEW."""
    findings = [
        Detection(entity_type="Crypto_Wallet_Seed", start=0, end=30, confidence=1.0, detector="test", explanation="Seed"),
        Detection(entity_type="Passport", start=35, end=45, confidence=1.0, detector="test", explanation="Passport"),
        Detection(entity_type="Biometric_Scan", start=50, end=65, confidence=1.0, detector="test", explanation="Fingerprint"),
    ]

    decisions = apply_policy(findings, mode="Personal")
    assert len(decisions) == 3

    for d in decisions:
        assert d.action == PolicyAction.REVIEW
        # Human readable reason explaining the fallback
        assert "strictly defaulting to REVIEW" in d.reason


def test_unknown_categories_default_to_review_developer():
    """Verify that categories not defined in Developer profile default to REVIEW."""
    # Aadhaar and Bank Account are defined in Personal, but NOT in Developer
    findings = [
        Detection(entity_type="Aadhaar", start=0, end=12, confidence=1.0, detector="test", explanation="Aadhaar"),
        Detection(entity_type="Bank Account", start=20, end=35, confidence=1.0, detector="test", explanation="Bank"),
    ]

    decisions = apply_policy(findings, mode="Developer")
    assert len(decisions) == 2
    for d in decisions:
        assert d.action == PolicyAction.REVIEW
        assert "defaulting to REVIEW" in d.reason


# =====================================================================
# 5. Normalization & Case-Insensitivity Tests
# =====================================================================

def test_policy_normalization_casing_and_separators():
    """Verify category matching handles lower/upper case and underscore/space variations."""
    findings = [
        Detection(entity_type="aadhaar", start=0, end=10, confidence=1.0, detector="test", explanation="lower"),
        Detection(entity_type="AADHAAR", start=15, end=25, confidence=1.0, detector="test", explanation="upper"),
        Detection(entity_type="bank_account", start=30, end=40, confidence=1.0, detector="test", explanation="snake"),
        Detection(entity_type="Bank Account", start=45, end=55, confidence=1.0, detector="test", explanation="space"),
    ]

    decisions = apply_policy(findings, mode="Personal")
    for d in decisions:
        assert d.action == PolicyAction.REDACT


def test_mode_name_case_insensitivity():
    """Verify mode name argument is case-insensitive."""
    findings = [
        Detection(entity_type="API_KEY", start=0, end=20, confidence=1.0, detector="test", explanation="key")
    ]

    decisions_lower = apply_policy(findings, mode="developer")
    assert decisions_lower[0].action == PolicyAction.BLOCK

    decisions_upper = apply_policy(findings, mode="DEVELOPER")
    assert decisions_upper[0].action == PolicyAction.BLOCK


# =====================================================================
# 6. Aggregate Document-Level Decision Tests
# =====================================================================

def test_aggregate_decision_precedence():
    """Verify document-level decision precedence: BLOCK > REVIEW > REDACT > ALLOW."""
    engine = PolicyEngine(mode="Developer")

    # 1. Empty findings -> ALLOW
    assert engine.get_aggregate_decision([]).action == PolicyAction.ALLOW

    # 2. Only ALLOW findings -> ALLOW
    allow_decision = PolicyDecision(entity_type="Email", action=PolicyAction.ALLOW, reason="Allowed")
    assert engine.get_aggregate_decision([allow_decision]).action == PolicyAction.ALLOW

    # 3. REDACT + ALLOW -> REDACT
    redact_decision = PolicyDecision(entity_type="Phone", action=PolicyAction.REDACT, reason="Redact")
    assert engine.get_aggregate_decision([allow_decision, redact_decision]).action == PolicyAction.REDACT

    # 4. REVIEW + REDACT + ALLOW -> REVIEW
    review_decision = PolicyDecision(entity_type="IP_ADDRESS", action=PolicyAction.REVIEW, reason="Review")
    assert engine.get_aggregate_decision([allow_decision, redact_decision, review_decision]).action == PolicyAction.REVIEW

    # 5. BLOCK + REVIEW + REDACT + ALLOW -> BLOCK (highest precedence)
    block_decision = PolicyDecision(entity_type="API_KEY", action=PolicyAction.BLOCK, reason="Block")
    assert engine.get_aggregate_decision(
        [allow_decision, redact_decision, review_decision, block_decision]
    ).action == PolicyAction.BLOCK


# =====================================================================
# 7. Edge Cases & Robustness
# =====================================================================

def test_empty_detections_list():
    """Verify empty input list returns empty decision list."""
    assert apply_policy([], mode="Personal") == []
    assert apply_policy([], mode="Developer") == []


def test_unsupported_mode_falls_back_safely():
    """Verify requesting an invalid profile logs warning and defaults to Personal."""
    findings = [
        Detection(entity_type="Aadhaar", start=0, end=10, confidence=1.0, detector="test", explanation="Aadhaar")
    ]
    decisions = apply_policy(findings, mode="NonExistentProfile123")
    assert len(decisions) == 1
    # Falls back to Personal -> REDACT
    assert decisions[0].action == PolicyAction.REDACT


def test_supported_modes_list():
    """Verify SUPPORTED_MODES constant lists Personal and Developer."""
    assert "Personal" in SUPPORTED_MODES
    assert "Developer" in SUPPORTED_MODES
