"""Core data models and domain types for SnapShield AI.

Ensures strict typing, immutability where appropriate, and architectural separation
between original content and sanitized safe output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported input document classifications."""
    RAW_TEXT = "RAW_TEXT"
    PDF = "PDF"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


class EntityCategory(str, Enum):
    """Categorization of sensitive information entities."""
    CREDENTIALS = "CREDENTIALS"           # API keys, passwords, private keys
    PAYMENT_CARD = "PAYMENT_CARD"         # Credit cards, debit cards, CVVs
    GOVERNMENT_ID = "GOVERNMENT_ID"       # SSN, Passport, Driver's License
    HEALTH_INFO = "HEALTH_INFO"           # Medical records, PHI, conditions
    CONTACT_INFO = "CONTACT_INFO"         # Emails, phone numbers, physical addresses
    INTERNAL_INFRA = "INTERNAL_INFRA"     # Internal IP addresses, hostnames, auth headers
    CUSTOM = "CUSTOM"


class DetectionSource(str, Enum):
    """Source that detected the sensitive finding."""
    DETERMINISTIC = "DETERMINISTIC"       # Regex, structured patterns
    CHECKSUM = "CHECKSUM"                 # Luhn, Mod-97, or verification algorithms
    LOCAL_OCR = "LOCAL_OCR"               # Local OCR text stream
    LOCAL_AI = "LOCAL_AI"                 # On-device contextual ONNX model
    HYBRID = "HYBRID"


class RiskLevel(str, Enum):
    """Qualitative risk score tier."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyAction(str, Enum):
    """Enforcement policy actions following risk assessment."""
    ALLOW = "ALLOW"                       # Safe as-is
    REDACT = "REDACT"                     # Mask/redact detected findings before forwarding
    REVIEW = "REVIEW"                     # Flag for manual user inspection and approval
    BLOCK = "BLOCK"                       # Outright rejection; egress strictly forbidden


class Finding(BaseModel):
    """Represents a specific sensitive entity discovered within inspected content."""
    entity_category: EntityCategory
    entity_type: str                      # e.g., "CREDIT_CARD_VISA", "SSN_US", "OPENAI_API_KEY"
    start_pos: int = Field(ge=0)
    end_pos: int = Field(ge=0)
    detection_source: DetectionSource
    confidence_score: float = Field(ge=0.0, le=1.0)
    checksum_verified: Optional[bool] = None
    mask_placeholder: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Note: We deliberately do NOT store raw unmasked sensitive text here
    # to maintain the zero-leakage invariant in logging and data transfers.


class RiskAssessment(BaseModel):
    """Composite risk evaluation derived from detected findings."""
    total_score: float = Field(ge=0.0, le=100.0)
    level: RiskLevel
    findings_count: int = Field(ge=0)
    category_breakdown: Dict[str, int] = Field(default_factory=dict)
    summary_reasons: List[str] = Field(default_factory=list)


class PolicyDecision(BaseModel):
    """Outcome of policy evaluation determining the fate of the content."""
    action: PolicyAction
    profile_applied: str
    risk_assessment: RiskAssessment
    enforced_at_utc: str
    rationale: str


class RedactionResult(BaseModel):
    """Encapsulates sanitized output while strictly decoupling it from original input."""
    sanitized_text: str
    redactions_applied: int
    findings_redacted: List[Finding]
    integrity_hash: str                  # SHA-256 hash of the sanitized text


class CheckpointResult(BaseModel):
    """Complete end-to-end outcome of a document passing through SnapShield AI."""
    session_id: str
    document_type: DocumentType
    detected_findings: List[Finding]
    risk_assessment: RiskAssessment
    policy_decision: PolicyDecision
    redaction_result: Optional[RedactionResult] = None
    external_llm_forwarded: bool = False
    external_llm_response: Optional[str] = None
    processing_time_ms: float
    hardware_acceleration_used: str     # e.g., "CPUExecutionProvider", "QNNExecutionProvider"
