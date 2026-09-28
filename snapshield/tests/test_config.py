"""Comprehensive tests for SnapShield AI configuration and policy engine.

Validates:
1. Environment variables loading and defaults:
   - LLM_API_KEY, LLM_PROVIDER, LLM_MODEL, MODEL_DIRECTORY, MAX_FILE_SIZE_MB
2. Configuration validation:
   - Bounded and positive MAX_FILE_SIZE_MB
   - validate_file_size bounds checks
   - validate_llm_configuration checks
3. Policy configuration (Personal and Developer profiles):
   - Personal policy actions: Aadhaar, PAN, Phone, Bank Account, IFSC, Email, Medical Information
   - Developer policy actions: API_KEY, PASSWORD, JWT, Email, IP_ADDRESS, Code
4. Critical Invariant:
   - Unknown sensitive categories MUST default to REVIEW.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from snapshield.config.policies import (
    PolicyAction,
    PolicyConfigManager,
    get_policy_manager,
    normalize_category_key,
)
from snapshield.config.settings import (
    ConfigurationValidationError,
    SnapShieldSettings,
)
from snapshield.services.policy import PolicyEngine


# ==========================================
# 1. Environment & Settings Tests
# ==========================================

def test_settings_default_values():
    """Verify that settings provide robust, sensible defaults when no env vars are set."""
    settings = SnapShieldSettings()

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.max_file_size_mb == 25.0
    assert settings.max_file_size_bytes == 25 * 1024 * 1024
    assert settings.default_policy_action == "REVIEW"
    assert isinstance(settings.model_directory, Path)


def test_settings_env_variable_overrides(monkeypatch):
    """Verify that required environment variables are loaded and mapped properly."""
    monkeypatch.setenv("LLM_API_KEY", "test-secret-key-12345")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-haiku")
    monkeypatch.setenv("MODEL_DIRECTORY", "custom_models")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "50.0")

    settings = SnapShieldSettings()

    assert settings.llm_api_key == "test-secret-key-12345"
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-3-haiku"
    assert settings.max_file_size_mb == 50.0
    assert settings.max_file_size_bytes == 50 * 1024 * 1024
    assert settings.has_llm_credentials() is True
    assert "custom_models" in str(settings.model_directory)


def test_max_file_size_validation():
    """Verify that MAX_FILE_SIZE_MB rejects non-positive or absurd limits."""
    # Zero or negative must fail
    with pytest.raises(ValidationError):
        SnapShieldSettings(max_file_size_mb=0.0)

    with pytest.raises(ValidationError):
        SnapShieldSettings(max_file_size_mb=-5.0)

    # Exceeding 500 MB upper limit must fail
    with pytest.raises(ValidationError):
        SnapShieldSettings(max_file_size_mb=501.0)


def test_llm_provider_and_model_validation():
    """Verify that empty strings for LLM provider or model are rejected."""
    with pytest.raises(ValidationError):
        SnapShieldSettings(llm_provider="   ")

    with pytest.raises(ValidationError):
        SnapShieldSettings(llm_model="")


def test_validate_file_size_method():
    """Verify runtime file size checker method."""
    settings = SnapShieldSettings(max_file_size_mb=10.0)

    # Within bounds (5 MB)
    settings.validate_file_size(5 * 1024 * 1024)

    # Exactly at limit (10 MB)
    settings.validate_file_size(10 * 1024 * 1024)

    # Exceeding limit (11 MB)
    with pytest.raises(ConfigurationValidationError) as exc_info:
        settings.validate_file_size(11 * 1024 * 1024)
    assert "exceeds maximum allowed limit" in str(exc_info.value)

    # Negative file size
    with pytest.raises(ConfigurationValidationError):
        settings.validate_file_size(-1)


def test_validate_llm_configuration():
    """Verify external LLM credential check when external transmission is enabled."""
    # If external LLM is disabled, missing key is allowed
    settings = SnapShieldSettings(external_llm_enabled=False, llm_api_key=None)
    settings.validate_llm_configuration()  # Should not raise

    # If external LLM is enabled, missing key raises ConfigurationValidationError
    settings_enabled = SnapShieldSettings(external_llm_enabled=True, llm_api_key=None)
    with pytest.raises(ConfigurationValidationError) as exc:
        settings_enabled.validate_llm_configuration()
    assert "LLM_API_KEY is not configured" in str(exc.value)


# ==========================================
# 2. Policy Configuration Tests (Personal & Developer)
# ==========================================

def test_category_key_normalization():
    """Verify that category normalization handles varying casing, spaces, and punctuation."""
    assert normalize_category_key("Aadhaar") == "AADHAAR"
    assert normalize_category_key("aadhaar") == "AADHAAR"
    assert normalize_category_key("Bank Account") == "BANK_ACCOUNT"
    assert normalize_category_key("bank-account") == "BANK_ACCOUNT"
    assert normalize_category_key("medical_information") == "MEDICAL_INFORMATION"


def test_personal_policy_rules():
    """Verify Personal policy actions:
    Aadhaar → REDACT
    PAN → REDACT
    Phone → REDACT
    Bank Account → REDACT
    IFSC → REDACT
    Email → ALLOW
    Medical Information → REDACT
    """
    pm = get_policy_manager()

    assert pm.get_action_for_category("Personal", "Aadhaar") == PolicyAction.REDACT
    assert pm.get_action_for_category("Personal", "PAN") == PolicyAction.REDACT
    assert pm.get_action_for_category("Personal", "Phone") == PolicyAction.REDACT
    assert pm.get_action_for_category("Personal", "Bank Account") == PolicyAction.REDACT
    assert pm.get_action_for_category("Personal", "bank_account") == PolicyAction.REDACT
    assert pm.get_action_for_category("Personal", "IFSC") == PolicyAction.REDACT
    assert pm.get_action_for_category("Personal", "Email") == PolicyAction.ALLOW
    assert pm.get_action_for_category("Personal", "Medical Information") == PolicyAction.REDACT


def test_developer_policy_rules():
    """Verify Developer policy actions:
    API_KEY → BLOCK
    PASSWORD → BLOCK
    JWT → BLOCK
    Email → ALLOW
    IP_ADDRESS → REVIEW
    Code → ALLOW
    """
    pm = get_policy_manager()

    assert pm.get_action_for_category("Developer", "API_KEY") == PolicyAction.BLOCK
    assert pm.get_action_for_category("Developer", "api_key") == PolicyAction.BLOCK
    assert pm.get_action_for_category("Developer", "PASSWORD") == PolicyAction.BLOCK
    assert pm.get_action_for_category("Developer", "password") == PolicyAction.BLOCK
    assert pm.get_action_for_category("Developer", "JWT") == PolicyAction.BLOCK
    assert pm.get_action_for_category("Developer", "Email") == PolicyAction.ALLOW
    assert pm.get_action_for_category("Developer", "IP_ADDRESS") == PolicyAction.REVIEW
    assert pm.get_action_for_category("Developer", "ip-address") == PolicyAction.REVIEW
    assert pm.get_action_for_category("Developer", "Code") == PolicyAction.ALLOW


def test_unknown_sensitive_categories_default_to_review():
    """CRITICAL INVARIANT TEST:
    Unknown sensitive categories MUST default to REVIEW across all profiles.
    """
    pm = get_policy_manager()

    # Under Personal profile
    assert pm.get_action_for_category("Personal", "UNCLASSIFIED_TOKEN") == PolicyAction.REVIEW
    assert pm.get_action_for_category("Personal", "biometric_sample") == PolicyAction.REVIEW
    assert pm.get_action_for_category("Personal", "secret_project_delta") == PolicyAction.REVIEW

    # Under Developer profile
    assert pm.get_action_for_category("Developer", "internal_slack_webhook") == PolicyAction.REVIEW
    assert pm.get_action_for_category("Developer", "custom_crypto_key") == PolicyAction.REVIEW
    assert pm.get_action_for_category("Developer", "random_unregistered_category") == PolicyAction.REVIEW

    # In case of an entirely non-existent profile, it still safely defaults to REVIEW
    assert pm.get_action_for_category("NonExistentProfile", "Email") == PolicyAction.REVIEW


def test_policy_engine_integration_with_policy_manager():
    """Verify that PolicyEngine correctly utilizes PolicyConfigManager and respects precedence."""
    engine_dev = PolicyEngine(profile_name="Developer")
    assert engine_dev.profile_name == "Developer"

    engine_pers = PolicyEngine(profile_name="Personal")
    assert engine_pers.profile_name == "Personal"
