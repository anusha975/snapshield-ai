"""Strongly typed data models for SnapShield AI.

Defines domain models for documents, OCR extraction, detections, policy decisions,
risk assessments, telemetry timings, benchmark results, and scan outputs.

Enforces:
- Rule 1: Never send original sensitive content to an external LLM.
- Rule 2: Never expose raw sensitive values through logs.
- Rule 3: Never fabricate AI confidence.
- Rule 8: Keep original input separate from sanitized output.
- Clean API with safe display helpers that mask sensitive values.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SeverityLevel(str, Enum):
    """Qualitative severity classification for detected findings."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyAction(str, Enum):
    """Authoritative policy enforcement actions."""
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class RiskLevel(str, Enum):
    """Aggregate risk tier."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ==========================================
# Safe Masking Helper Functions
# ==========================================

def mask_sensitive_value(value: str, entity_type: str = "GENERIC") -> str:
    """Masks a sensitive string value for safe display and logging.

    Preserves formatting cues (e.g., last 4 digits of a card or domain of an email)
    while thoroughly obfuscating the confidential component.
    """
    if not value:
        return "[EMPTY]"

    val = value.strip()
    entity_upper = entity_type.upper()

    # Credit card / Payment card masking (keep last 4)
    if "CARD" in entity_upper or "PAYMENT" in entity_upper:
        digits = [c for c in val if c.isdigit()]
        if len(digits) >= 4:
            return f"****-****-****-{''.join(digits[-4:])}"
        return "[REDACTED_CARD]"

    # US SSN or National ID (keep last 4)
    if "SSN" in entity_upper or "AADHAAR" in entity_upper or "PAN" in entity_upper:
        if len(val) >= 4:
            return f"***-**-{val[-4:]}"
        return "[REDACTED_GOV_ID]"

    # Email address (mask local part, keep domain)
    if "EMAIL" in entity_upper and "@" in val:
        local, domain = val.split("@", 1)
        masked_local = local[0] + "***" if len(local) > 1 else "***"
        return f"{masked_local}@{domain}"

    # Secrets, API Keys, Passwords, Tokens (mask everything)
    if any(k in entity_upper for k in ("KEY", "SECRET", "PASSWORD", "JWT", "TOKEN")):
        prefix = val[:3] if len(val) >= 8 else ""
        return f"{prefix}***[REDACTED_SECRET]***"

    # Default fallback: partial mask
    if len(val) <= 4:
        return "***"
    return f"{val[:2]}***{val[-2:]}"


# ==========================================
# Domain Models
# ==========================================

class InputDocument(BaseModel):
    """Represents a document or text payload ingested into SnapShield."""
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_type: str = Field(..., description="Document format: RAW_TEXT, PDF, IMAGE, etc.")
    raw_content: Optional[str] = Field(default=None, repr=False, description="Raw content, excluded from default repr.")
    raw_bytes: Optional[bytes] = Field(default=None, repr=False, description="Raw binary content, excluded from default repr.")
    file_name: Optional[str] = None
    file_size_bytes: int = Field(ge=0, default=0)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def safe_summary(self) -> Dict[str, Any]:
        """Returns non-sensitive metadata safe for logging and debugging."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "created_at_utc": self.created_at_utc,
            "content_length_chars": len(self.raw_content) if self.raw_content else 0,
        }

    def __repr__(self) -> str:
        return (
            f"InputDocument(id={self.document_id}, type={self.document_type}, "
            f"name={self.file_name}, size={self.file_size_bytes}B)"
        )


class OCRResult(BaseModel):
    """Optical Character Recognition extraction outcome with confidence metrics."""
    text: str = Field(..., repr=False, description="Extracted text from image or screenshot.")
    confidence: float = Field(ge=0.0, le=1.0, description="Truthful average confidence (0.0 to 1.0).")
    engine: str = Field(default="tesseract_fallback", description="OCR engine identifier.")
    word_count: int = Field(ge=0, default=0)
    char_count: int = Field(ge=0, default=0)
    processing_time_ms: float = Field(ge=0.0, default=0.0)
    is_successful: bool = True
    error_message: Optional[str] = None

    def safe_summary(self) -> Dict[str, Any]:
        """Provides telemetry and status without emitting extracted text into logs."""
        return {
            "engine": self.engine,
            "confidence": round(self.confidence, 4),
            "word_count": self.word_count,
            "char_count": self.char_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "is_successful": self.is_successful,
            "error_message": self.error_message,
        }


class Detection(BaseModel):
    """Specific sensitive entity detected within inspected text.

    Enforces Rule 2: Never expose raw sensitive values through logs.
    """
    entity_type: str = Field(..., description="Classification category (e.g. API_KEY, PAYMENT_CARD, SSN).")
    start: int = Field(ge=0, description="Start character offset in original text.")
    end: int = Field(ge=0, description="End character offset in original text.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score (0.0 to 1.0).")
    detector: str = Field(..., description="Detector source (e.g. regex, luhn_checksum, local_onnx).")
    valid: bool = Field(default=True, description="Algorithmic validation status (e.g. Luhn pass).")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Finding severity level.")
    explanation: str = Field(..., description="Reasoning or description for this detection.")
    raw_value: Optional[str] = Field(default=None, repr=False, description="Raw sensitive value. NEVER logged.")

    @field_validator("end")
    @classmethod
    def validate_span(cls, end_val: int, info) -> int:
        start_val = info.data.get("start", 0)
        if end_val < start_val:
            raise ValueError(f"Span end ({end_val}) cannot be strictly less than start ({start_val}).")
        return end_val

    @property
    def type(self) -> str:
        """Alias property matching detection type specification."""
        return self.entity_type

    def get_masked_display_value(self) -> str:
        """Returns safe, masked display representation for UI and logs."""
        if not self.raw_value:
            return f"[REDACTED_{self.entity_type}]"
        return mask_sensitive_value(self.raw_value, self.entity_type)

    def safe_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary guaranteed free of raw sensitive strings."""
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
            "detector": self.detector,
            "valid": self.valid,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "masked_value": self.get_masked_display_value(),
        }

    def __repr__(self) -> str:
        return (
            f"Detection(entity={self.entity_type}, span=[{self.start}:{self.end}], "
            f"conf={self.confidence:.2f}, detector={self.detector}, "
            f"valid={self.valid}, severity={self.severity.value})"
        )


class PolicyDecision(BaseModel):
    """Enforcement decision evaluating a sensitive entity or overall scan."""
    entity_type: str = Field(..., description="Target sensitive category or '*' for overall document.")
    action: PolicyAction = Field(..., description="Policy action: ALLOW, REDACT, BLOCK, REVIEW.")
    reason: str = Field(..., description="Deterministic justification for the policy decision.")
    profile_applied: Optional[str] = Field(default=None, description="Profile name (e.g. Personal, Developer).")


class RiskAssessment(BaseModel):
    """Aggregated risk score and breakdown derived from detected findings."""
    score: float = Field(ge=0.0, le=100.0, description="Risk score on 0.0 to 100.0 scale.")
    level: RiskLevel = Field(default=RiskLevel.LOW, description="Qualitative risk tier.")
    findings_count: int = Field(ge=0, default=0, description="Total number of findings.")
    category_breakdown: Dict[str, int] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    is_blocked: bool = Field(default=False, description="Flag indicating if risk mandates outright blocking.")


class ScanTiming(BaseModel):
    """Telemetry timings for each phase of the inspection pipeline (in milliseconds)."""
    extraction_ms: float = Field(ge=0.0, default=0.0)
    ocr_ms: float = Field(ge=0.0, default=0.0)
    structured_detection_ms: float = Field(ge=0.0, default=0.0)
    ai_inference_ms: float = Field(ge=0.0, default=0.0)
    detection_ms: float = Field(ge=0.0, default=0.0)
    risk_ms: float = Field(ge=0.0, default=0.0)
    risk_assessment_ms: float = Field(ge=0.0, default=0.0)
    policy_ms: float = Field(ge=0.0, default=0.0)
    redaction_ms: float = Field(ge=0.0, default=0.0)
    total_ms: float = Field(ge=0.0, default=0.0)

    def calculate_total(self) -> float:
        """Sums component timings to update total_ms."""
        det_total = self.structured_detection_ms + self.ai_inference_ms if (self.structured_detection_ms or self.ai_inference_ms) else self.detection_ms
        risk_total = self.risk_ms or self.risk_assessment_ms
        self.detection_ms = round(det_total, 2)
        self.risk_assessment_ms = round(risk_total, 2)
        self.risk_ms = round(risk_total, 2)
        self.total_ms = round(
            self.extraction_ms
            + self.ocr_ms
            + det_total
            + risk_total
            + self.policy_ms
            + self.redaction_ms,
            2,
        )
        return self.total_ms



class RuntimeStatus(BaseModel):
    """Hardware diagnostic telemetry confirming actual execution environment without fabrication."""
    os_name: str
    machine_arch: str
    processor: str
    active_execution_provider: str
    is_accelerated: bool
    device_backend: str = Field(..., description="e.g. Snapdragon NPU (QNN), DirectML, or Verified CPU Fallback.")
    memory_usage_mb: Optional[float] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    onnxruntime_version: str = "unavailable"
    available_providers: List[str] = Field(default_factory=list)
    mode: str = "AUTO"
    is_npu_active: bool = False


class BenchmarkResult(BaseModel):
    """Performance telemetry recording local AI inference or pipeline execution benchmarks."""
    benchmark_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    execution_provider: str
    sample_name: str
    iterations: int = Field(ge=0)
    mean_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    throughput_items_per_sec: float = Field(ge=0.0)
    first_inference_ms: Optional[float] = None
    warm_inference_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    std_dev_ms: Optional[float] = None
    is_accelerated: bool = False
    status_message: Optional[str] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hardware_details: Dict[str, str] = Field(default_factory=dict)


class ScanResult(BaseModel):
    """Complete end-to-end outcome of document analysis and policy enforcement.

    Strictly satisfies:
    - Rule 1: Never send original sensitive content to an external LLM.
    - Rule 2: Never expose raw sensitive values through logs.
    - Rule 8: Keep original input separate from sanitized output.
    """
    scan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_text: str = Field(..., repr=False, description="Original unredacted text. EXCLUDED from default repr & safe logs.")
    sanitized_text: str = Field(..., description="Redacted / safe output decoupled from original text.")
    detections: List[Detection] = Field(default_factory=list)
    decisions: List[PolicyDecision] = Field(default_factory=list)
    risk_assessment: RiskAssessment
    timings: ScanTiming
    runtime_status: RuntimeStatus
    errors: List[str] = Field(default_factory=list)
    integrity_hash: str = Field(default="", description="SHA-256 hash of sanitized text.")
    ocr_bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    annotated_image_bytes: Optional[bytes] = Field(default=None, repr=False)
    annotated_regions_count: int = 0

    def model_post_init(self, __context) -> None:
        """Computes SHA-256 integrity hash of sanitized text if not supplied."""
        if not self.integrity_hash and self.sanitized_text is not None:
            self.integrity_hash = hashlib.sha256(self.sanitized_text.encode("utf-8")).hexdigest()

    @property
    def is_blocked(self) -> bool:
        """Returns True if any policy decision triggered PolicyAction.BLOCK."""
        return any(d.action == PolicyAction.BLOCK for d in self.decisions) or self.risk_assessment.is_blocked

    @property
    def overall_action(self) -> PolicyAction:
        """Determines the dominant action across all policy decisions."""
        if any(d.action == PolicyAction.BLOCK for d in self.decisions):
            return PolicyAction.BLOCK
        if any(d.action == PolicyAction.REVIEW for d in self.decisions):
            return PolicyAction.REVIEW
        if any(d.action == PolicyAction.REDACT for d in self.decisions):
            return PolicyAction.REDACT
        return PolicyAction.ALLOW

    def safe_summary(self) -> Dict[str, Any]:
        """Provides a safe audit summary for logging without exposing original raw content."""
        return {
            "scan_id": self.scan_id,
            "overall_action": self.overall_action.value,
            "is_blocked": self.is_blocked,
            "findings_count": len(self.detections),
            "detections": [d.safe_dict() for d in self.detections],
            "risk_score": self.risk_assessment.score,
            "risk_level": self.risk_assessment.level.value,
            "total_ms": self.timings.total_ms,
            "hardware_backend": self.runtime_status.device_backend,
            "integrity_hash": self.integrity_hash,
            "errors": self.errors,
            "sanitized_length": len(self.sanitized_text),
            "original_length": len(self.original_text),
        }

    def __repr__(self) -> str:
        return (
            f"ScanResult(id={self.scan_id}, action={self.overall_action.value}, "
            f"blocked={self.is_blocked}, detections={len(self.detections)}, "
            f"risk_score={self.risk_assessment.score}, time_ms={self.timings.total_ms})"
        )
