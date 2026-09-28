"""Deterministic source-code and credential detection engine for SnapShield AI.

Detects high-risk developer credentials and code leakages without an LLM:
1. API Keys (OpenAI, GitHub, Slack, generic tokens)
2. Password assignments in code and configuration
3. JWT-like tokens (three-part Base64URL header.payload.signature)
4. AWS-style credentials (AKIA/ASIA access keys, 40-char secret keys)
5. Bearer authentication tokens
6. Credential-containing URLs (protocol://user:password@host)
7. Private key markers (RSA, EC, DSA, OpenSSH, PGP)
8. Database connection strings (PostgreSQL, MongoDB, MySQL, Redis, JDBC)

Enforces:
- Severity is HIGH or CRITICAL.
- Never prints or logs matched raw secret contents.
- Masks previews only.
- Seamless compatibility with the Detection domain model and PolicyEngine.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from snapshield.core.logger import logger
from snapshield.core.models import Detection, SeverityLevel


# ==========================================
# Regular Expression Patterns
# ==========================================

# 1. AWS Credentials
RE_AWS_ACCESS_KEY = re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b")
RE_AWS_SECRET_KEY = re.compile(
    r"(?i)\b(?:aws_secret_access_key|aws_secret|secret_access_key)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?\b"
)

# 2. API Keys
RE_OPENAI_KEY = re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9-_]{20,56}\b")
RE_GITHUB_TOKEN = re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{50,82}\b")
RE_SLACK_TOKEN = re.compile(r"\bxox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24,32}\b")
RE_GENERIC_API_KEY = re.compile(
    r"""(?i)["']?(?:[a-zA-Z0-9_\-\.]*(?:api[_-]?key|client[_-]?secret|auth[_-]?token|app[_-]?secret)[a-zA-Z0-9_\-\.]*)["']?\s*[:=]\s*["']?([A-Za-z0-9_\-\.]{16,64})["']?"""
)

# 3. Password Assignments (handles YAML, env, JSON, and prefixed vars like POSTGRES_PASSWORD)
RE_PASSWORD_ASSIGNMENT = re.compile(
    r"""(?i)["']?(?:[a-zA-Z0-9_\-\.]*(?:password|passwd|pwd|db_pass)[a-zA-Z0-9_\-\.]*)["']?\s*[:=]\s*["']?([^"'\s\r\n]{4,64})["']?"""
)

# 4. JWT-like Tokens (Header.Payload.Signature with Base64URL eyJ...)
RE_JWT = re.compile(r"\beyJ[A-Za-z0-9-_]{10,}\.eyJ[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\b")

# 5. Bearer Tokens
RE_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-\.=]{16,})\b")

# 6. Credential-Containing Web URLs (HTTP/HTTPS/FTP/SFTP)
RE_CREDENTIAL_URL = re.compile(
    r"\b(?:https?|ftp|sftp)://([^:@\s]{2,}):([^@\s]{2,})@([A-Za-z0-9_\-\.]+)(?::\d+)?(?:/[^\s]*)?\b"
)

# 7. Private Key Markers
RE_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----|-----BEGIN PGP PRIVATE KEY BLOCK-----"
)

# 8. Database Connection Strings (PostgreSQL, MongoDB, MySQL, Redis, JDBC, ODBC)
RE_DB_CONN_URI = re.compile(
    r"(?i)\b(?:postgresql|postgres|mongodb(?:\+srv)?|mysql|redis)://(?:[A-Za-z0-9_\-\.%]+:[^@\s\r\n]+@)[A-Za-z0-9_\-\.]+(?::\d+)?(?:/[A-Za-z0-9_\-\.]*)?\b"
)
RE_JDBC_OR_ODBC = re.compile(
    r"(?i)\b(?:Server|Data Source)=[^;\r\n]+;Database=[^;\r\n]+;(?:Uid|User Id)=[^;\r\n]+;(?:Pwd|Password)=[^;\r\n]+"
)


# ==========================================
# Detection Function
# ==========================================

def detect_credentials(text: str) -> list[Detection]:
    """Scans code, config, or prompt text for secrets and credentials.

    Enforces:
    - Purely deterministic pattern matching (zero LLM calls).
    - HIGH and CRITICAL severity classifications.
    - Zero exposure of raw secret content in logs.
    - Non-overlapping span protection.
    """
    if not text or not isinstance(text, str):
        return []

    findings: list[Detection] = []
    occupied_spans: list[Tuple[int, int]] = []

    def _is_overlapping(start: int, end: int) -> bool:
        for s, e in occupied_spans:
            if max(start, s) < min(end, e):
                return True
        return False

    def _add_detection(
        entity_type: str,
        start: int,
        end: int,
        confidence: float,
        detector: str,
        severity: SeverityLevel,
        explanation: str,
        raw_val: str,
    ) -> None:
        if _is_overlapping(start, end):
            return
        findings.append(
            Detection(
                entity_type=entity_type,
                start=start,
                end=end,
                confidence=confidence,
                detector=detector,
                valid=True,
                severity=severity,
                explanation=explanation,
                raw_value=raw_val,
            )
        )
        occupied_spans.append((start, end))

    # 1. Private Keys (CRITICAL)
    for match in RE_PRIVATE_KEY.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="PRIVATE_KEY",
            start=start,
            end=end,
            confidence=1.0,
            detector="regex_private_key",
            severity=SeverityLevel.CRITICAL,
            explanation="Cryptographic private key header marker detected.",
            raw_val=match.group(0),
        )

    # 2. Credential-Containing URLs (CRITICAL)
    for match in RE_CREDENTIAL_URL.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="CREDENTIAL_URL",
            start=start,
            end=end,
            confidence=0.98,
            detector="regex_credential_url",
            severity=SeverityLevel.CRITICAL,
            explanation="Network URL containing embedded plaintext username and password credentials.",
            raw_val=match.group(0),
        )

    # 3. Database Connection Strings (CRITICAL)
    for match in RE_DB_CONN_URI.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="DATABASE_URI",
            start=start,
            end=end,
            confidence=0.98,
            detector="regex_database_uri",
            severity=SeverityLevel.CRITICAL,
            explanation="Database connection URI containing authentication credentials.",
            raw_val=match.group(0),
        )

    for match in RE_JDBC_OR_ODBC.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="DATABASE_URI",
            start=start,
            end=end,
            confidence=0.98,
            detector="regex_jdbc_odbc",
            severity=SeverityLevel.CRITICAL,
            explanation="Database connection string with explicit password parameters.",
            raw_val=match.group(0),
        )

    # 4. AWS Credentials (CRITICAL)
    for match in RE_AWS_ACCESS_KEY.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="AWS_ACCESS_KEY",
            start=start,
            end=end,
            confidence=0.95,
            detector="regex_aws_key",
            severity=SeverityLevel.CRITICAL,
            explanation="Amazon Web Services (AWS) 20-character Access Key ID.",
            raw_val=match.group(0),
        )

    for match in RE_AWS_SECRET_KEY.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="AWS_SECRET_KEY",
            start=start,
            end=end,
            confidence=0.95,
            detector="regex_aws_secret",
            severity=SeverityLevel.CRITICAL,
            explanation="Amazon Web Services (AWS) 40-character Secret Access Key.",
            raw_val=match.group(0),
        )

    # 5. JWT-like Tokens (HIGH)
    for match in RE_JWT.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="JWT",
            start=start,
            end=end,
            confidence=0.95,
            detector="regex_jwt",
            severity=SeverityLevel.HIGH,
            explanation="JSON Web Token (JWT) matching Base64URL header.payload.signature structure.",
            raw_val=match.group(0),
        )

    # 6. Specific API Keys: OpenAI, GitHub, Slack (CRITICAL)
    for match in RE_OPENAI_KEY.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="API_KEY",
            start=start,
            end=end,
            confidence=0.99,
            detector="regex_openai_key",
            severity=SeverityLevel.CRITICAL,
            explanation="OpenAI API authentication token.",
            raw_val=match.group(0),
        )

    for match in RE_GITHUB_TOKEN.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="API_KEY",
            start=start,
            end=end,
            confidence=0.99,
            detector="regex_github_token",
            severity=SeverityLevel.CRITICAL,
            explanation="GitHub Personal Access Token (PAT) / OAuth secret.",
            raw_val=match.group(0),
        )

    for match in RE_SLACK_TOKEN.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="API_KEY",
            start=start,
            end=end,
            confidence=0.99,
            detector="regex_slack_token",
            severity=SeverityLevel.CRITICAL,
            explanation="Slack bot / API authentication token.",
            raw_val=match.group(0),
        )

    # 7. Password Assignments (CRITICAL)
    for match in RE_PASSWORD_ASSIGNMENT.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="PASSWORD",
            start=start,
            end=end,
            confidence=0.90,
            detector="regex_password_assignment",
            severity=SeverityLevel.CRITICAL,
            explanation="Hardcoded password assignment in source code or configuration.",
            raw_val=match.group(0),
        )

    # 8. Bearer Tokens (HIGH)
    for match in RE_BEARER_TOKEN.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="BEARER_TOKEN",
            start=start,
            end=end,
            confidence=0.92,
            detector="regex_bearer_token",
            severity=SeverityLevel.HIGH,
            explanation="HTTP Bearer authentication token.",
            raw_val=match.group(0),
        )

    # 9. Generic API Key Assignments (HIGH)
    for match in RE_GENERIC_API_KEY.finditer(text):
        start, end = match.span()
        _add_detection(
            entity_type="API_KEY",
            start=start,
            end=end,
            confidence=0.88,
            detector="regex_generic_secret",
            severity=SeverityLevel.HIGH,
            explanation="Sensitive API key or client secret parameter assignment.",
            raw_val=match.group(0),
        )

    findings.sort(key=lambda d: d.start)

    # Never log secret values
    logger.info("Credential scan complete: detected %d secret entity/entities: %s",
                len(findings), [f.entity_type for f in findings])

    return findings
