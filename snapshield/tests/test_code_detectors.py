"""Comprehensive tests for source-code, credential, and secret detection.

Validates:
1. Detection patterns:
   - API keys (OpenAI, GitHub, Slack, generic)
   - Password assignments
   - JWT-like tokens
   - AWS-style credentials (AKIA/ASIA, secret keys)
   - Bearer tokens
   - Credential-containing URLs
   - Private key markers
   - Database connection strings
2. Severity enforcement:
   - Normally HIGH or CRITICAL
3. Privacy rules:
   - Never print raw matched secret contents
   - Masked previews only
4. Tests with synthetic developer configuration files
"""

from __future__ import annotations

from pathlib import Path
import pytest

from snapshield.core.code_detectors import detect_credentials
from snapshield.core.models import Detection, SeverityLevel


# ==========================================
# 1. API Keys & Token Tests
# ==========================================

def test_detect_openai_api_key():
    """Verify OpenAI secret key detection."""
    text = "export OPENAI_API_KEY=\"sk-proj-syntheticKey1234567890abcdefABCDEF12345\""
    findings = detect_credentials(text)

    api_findings = [f for f in findings if f.entity_type == "API_KEY"]
    assert len(api_findings) == 1
    f = api_findings[0]
    assert f.severity == SeverityLevel.CRITICAL
    assert f.confidence >= 0.95
    # Verify masked preview
    masked = f.get_masked_display_value()
    assert "[REDACTED_SECRET]" in masked
    assert "syntheticKey1234567890" not in masked


def test_detect_github_token():
    """Verify GitHub Personal Access Token (PAT) detection."""
    pfx = "ghp"
    text = f"const GITHUB_TOKEN = '{pfx}_syntheticGitHubTokenForTesting123456';"
    findings = detect_credentials(text)

    gh_findings = [f for f in findings if f.entity_type == "API_KEY"]
    assert len(gh_findings) == 1
    assert gh_findings[0].severity == SeverityLevel.CRITICAL


def test_detect_slack_bot_token():
    """Verify Slack bot token detection."""
    pfx = "xoxb"
    text = f"SLACK_BOT_TOKEN={pfx}-123456789012-123456789012-syntheticSlackTokenAlphaBeta"
    findings = detect_credentials(text)

    slack_findings = [f for f in findings if f.entity_type == "API_KEY"]
    assert len(slack_findings) == 1
    assert slack_findings[0].severity == SeverityLevel.CRITICAL


def test_detect_jwt_token():
    """Verify 3-part Base64URL JWT token detection."""
    jwt_sample = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlN5bnRoZXRpYyBVc2VyIn0."
        "syntheticSignaturePartHere1234567890abcdef"
    )
    text = f"Authorization header: {jwt_sample}"
    findings = detect_credentials(text)

    jwt_findings = [f for f in findings if f.entity_type == "JWT"]
    assert len(jwt_findings) == 1
    assert jwt_findings[0].severity == SeverityLevel.HIGH
    assert jwt_findings[0].confidence >= 0.95


# ==========================================
# 2. AWS Credentials Tests
# ==========================================

def test_detect_aws_credentials():
    """Verify AWS Access Key ID and Secret Access Key detection."""
    text = (
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    findings = detect_credentials(text)

    types = {f.entity_type for f in findings}
    assert "AWS_ACCESS_KEY" in types
    assert "AWS_SECRET_KEY" in types

    for f in findings:
        assert f.severity == SeverityLevel.CRITICAL
        assert f.confidence >= 0.95
        # Masking verification
        assert "EXAMPLEKEY" not in f.get_masked_display_value()


# ==========================================
# 3. Password Assignment Tests
# ==========================================

def test_detect_password_assignment():
    """Verify hardcoded password assignments in code/config."""
    text = "user_password = 'SuperSecureDeveloperPassword999!'"
    findings = detect_credentials(text)

    pwd_findings = [f for f in findings if f.entity_type == "PASSWORD"]
    assert len(pwd_findings) == 1
    f = pwd_findings[0]
    assert f.severity == SeverityLevel.CRITICAL
    assert "SuperSecure" not in f.get_masked_display_value()
    assert "[REDACTED_SECRET]" in f.get_masked_display_value()


# ==========================================
# 4. Bearer Token Tests
# ==========================================

def test_detect_bearer_token():
    """Verify HTTP Bearer token header detection."""
    text = "Authorization: Bearer secretBearerAuthTokenString12345678"
    findings = detect_credentials(text)

    bearer_findings = [f for f in findings if f.entity_type == "BEARER_TOKEN"]
    assert len(bearer_findings) == 1
    assert bearer_findings[0].severity == SeverityLevel.HIGH


# ==========================================
# 5. Credential-Containing URLs & Database Strings
# ==========================================

def test_detect_credential_containing_url():
    """Verify URL with embedded username and password."""
    text = "Webhook endpoint: https://admin_bot:SecretPassword999@internal.api.network/v1"
    findings = detect_credentials(text)

    url_findings = [f for f in findings if f.entity_type == "CREDENTIAL_URL"]
    assert len(url_findings) == 1
    assert url_findings[0].severity == SeverityLevel.CRITICAL


def test_detect_database_connection_uri():
    """Verify PostgreSQL / MongoDB / Redis connection URI with credentials."""
    text = "DATABASE_URL = postgres://db_user:RootPassword123@10.0.1.50:5432/production_db"
    findings = detect_credentials(text)

    db_findings = [f for f in findings if f.entity_type == "DATABASE_URI"]
    assert len(db_findings) == 1
    assert db_findings[0].severity == SeverityLevel.CRITICAL


def test_detect_jdbc_or_odbc_connection_string():
    """Verify JDBC/ODBC connection strings with explicit Password parameter."""
    text = "Conn = Server=db.server.internal;Database=app_db;Uid=admin;Pwd=MySecretSqlPass888;"
    findings = detect_credentials(text)

    db_findings = [f for f in findings if f.entity_type == "DATABASE_URI"]
    assert len(db_findings) == 1
    assert db_findings[0].severity == SeverityLevel.CRITICAL


# ==========================================
# 6. Private Key Marker Tests
# ==========================================

def test_detect_private_key_marker():
    """Verify RSA / OpenSSH private key block detection."""
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0syntheticKeyBlockForTestingOnlySnapShieldAI00000\n"
        "-----END RSA PRIVATE KEY-----"
    )
    findings = detect_credentials(text)

    key_findings = [f for f in findings if f.entity_type == "PRIVATE_KEY"]
    assert len(key_findings) == 1
    assert key_findings[0].severity == SeverityLevel.CRITICAL


# ==========================================
# 7. Synthetic Developer Configuration Files Tests
# ==========================================

def test_scan_synthetic_env_file():
    """Verify detection on synthetic app_config.env file."""
    env_path = Path(__file__).resolve().parent.parent / "data" / "developer_samples" / "app_config.env"
    assert env_path.exists()

    content = env_path.read_text(encoding="utf-8")
    findings = detect_credentials(content)

    detected_types = {f.entity_type for f in findings}
    assert "API_KEY" in detected_types
    assert "AWS_ACCESS_KEY" in detected_types
    assert "AWS_SECRET_KEY" in detected_types
    assert "DATABASE_URI" in detected_types
    assert "JWT" in detected_types
    assert "PASSWORD" in detected_types

    # Ensure all findings have HIGH or CRITICAL severity
    for f in findings:
        assert f.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)


def test_scan_synthetic_docker_compose_file():
    """Verify detection on synthetic docker-compose.yml file."""
    compose_path = Path(__file__).resolve().parent.parent / "data" / "developer_samples" / "docker-compose.yml"
    assert compose_path.exists()

    content = compose_path.read_text(encoding="utf-8")
    findings = detect_credentials(content)

    detected_types = {f.entity_type for f in findings}
    assert "PASSWORD" in detected_types
    assert "BEARER_TOKEN" in detected_types
    assert "AWS_ACCESS_KEY" in detected_types


def test_scan_synthetic_server_settings_json():
    """Verify detection on synthetic server_settings.json file."""
    json_path = Path(__file__).resolve().parent.parent / "data" / "developer_samples" / "server_settings.json"
    assert json_path.exists()

    content = json_path.read_text(encoding="utf-8")
    findings = detect_credentials(content)

    detected_types = {f.entity_type for f in findings}
    assert "API_KEY" in detected_types
    assert "JWT" in detected_types
    assert "CREDENTIAL_URL" in detected_types
    assert "DATABASE_URI" in detected_types


def test_scan_synthetic_private_key_file():
    """Verify detection on synthetic synthetic_private_key.pem file."""
    pem_path = Path(__file__).resolve().parent.parent / "data" / "developer_samples" / "synthetic_private_key.pem"
    assert pem_path.exists()

    content = pem_path.read_text(encoding="utf-8")
    findings = detect_credentials(content)

    assert len(findings) == 1
    assert findings[0].entity_type == "PRIVATE_KEY"
    assert findings[0].severity == SeverityLevel.CRITICAL


def test_hybrid_detection_engine_scan_detections_unified():
    """Verify HybridDetectionEngine.scan_detections merges credentials and PII seamlessly."""
    from snapshield.services.detector import HybridDetectionEngine

    mixed_text = (
        "Developer token: sk-proj-syntheticKeyAlpha1234567890abcdef\n"
        "Contact email: developer@example-domain.internal\n"
        "Database: postgres://admin:SecretPass999@10.0.0.1:5432/app"
    )
    engine = HybridDetectionEngine()
    detections = engine.scan_detections(mixed_text)

    types = {d.entity_type for d in detections}
    assert "API_KEY" in types
    assert "DATABASE_URI" in types
    assert "Email" in types
    # All are instances of Detection model
    for d in detections:
        assert isinstance(d, Detection)
        assert d.valid is True
        # Verify no raw secrets leaked in string representation
        assert "SecretPass999" not in str(d)
        assert "syntheticKeyAlpha" not in str(d)


def test_masked_previews_only_never_expose_secrets():
    """Verify that get_masked_display_value and repr never expose raw sensitive tokens."""
    text = "password = 'TopSecretInternalPassword123!'"
    findings = detect_credentials(text)

    assert len(findings) == 1
    f = findings[0]
    # Check masked preview
    masked = f.get_masked_display_value()
    assert "TopSecretInternalPassword123!" not in masked
    assert "[REDACTED_SECRET]" in masked
    # Check model repr
    rep = repr(f)
    assert "TopSecretInternalPassword123!" not in rep

