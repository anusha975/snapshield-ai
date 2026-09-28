"""Domain exceptions and error classification for SnapShield AI.

Provides structured, explicit error classes across the inspection and policy pipeline.
Enforces rule 14: Do not silently swallow exceptions.
Enforces graceful degradation:
- AI unavailable -> deterministic detection continues
- Acceleration unavailable -> CPU inference continues
- LLM unavailable -> sanitized output remains usable
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class SnapShieldError(Exception):
    """Base exception for all SnapShield AI domain errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or "ERR_SNAPSHIELD_GENERIC"


# =========================================================================
# Input & File Ingestion Exceptions
# =========================================================================

class InputValidationError(SnapShieldError):
    """Raised when an uploaded document or text payload fails security/integrity validation."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "ERR_INPUT_VALIDATION") -> None:
        super().__init__(message, details, error_code)


class EmptyFileError(InputValidationError):
    """Raised when an uploaded file or text payload is empty (0 bytes or whitespace)."""
    def __init__(self, message: str = "Input document is empty (0 bytes). Please provide content to analyze.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_EMPTY_FILE")


class OversizedFileError(InputValidationError):
    """Raised when an uploaded file exceeds the configured maximum allowable size limit."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_OVERSIZED_FILE")


class InvalidFileFormatError(InputValidationError):
    """Raised when an unsupported file extension or mismatched MIME magic header is detected."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_INVALID_FILE_FORMAT")


# =========================================================================
# Extraction & OCR Exceptions
# =========================================================================

class ExtractionError(SnapShieldError):
    """Raised when text or document parsing fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "ERR_EXTRACTION_FAILED") -> None:
        super().__init__(message, details, error_code)


class PDFExtractionError(ExtractionError):
    """Raised when PDF structure parsing, decompression, or rendering fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_PDF_EXTRACTION_FAILED")


class OCRError(SnapShieldError):
    """Raised when local optical character recognition fails or is misconfigured."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "ERR_OCR_FAILED") -> None:
        super().__init__(message, details, error_code)


class TesseractNotFoundError(OCRError):
    """Raised when local Tesseract binary is absent from system PATH."""
    def __init__(self, message: str = "Local Tesseract OCR binary not found in system PATH.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_TESSERACT_NOT_FOUND")


# =========================================================================
# Detection & AI Runtime Exceptions
# =========================================================================

class DetectionError(SnapShieldError):
    """Raised when detection engine encounters an unrecoverable scanning failure."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "ERR_DETECTION_FAILED") -> None:
        super().__init__(message, details, error_code)


class ModelUnavailableError(DetectionError):
    """Raised when local ONNX model file is absent from configured model directory."""
    def __init__(self, message: str = "Local ONNX model file is not available.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_MODEL_UNAVAILABLE")


class MalformedModelError(DetectionError):
    """Raised when local ONNX model graph or weights are corrupted or incompatible."""
    def __init__(self, message: str = "Local ONNX model file is corrupted or malformed.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_MALFORMED_MODEL")


class RuntimeUnavailableError(SnapShieldError):
    """Raised when ONNX runtime or required backend binary is unavailable on the host."""
    def __init__(self, message: str = "ONNX runtime environment is unavailable.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_RUNTIME_UNAVAILABLE")


class HardwareAccelerationError(SnapShieldError):
    """Raised when requested hardware acceleration (e.g. QNN/NPU) cannot be initialized."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "ERR_HARDWARE_ACCELERATION") -> None:
        super().__init__(message, details, error_code)


class AcceleratedProviderError(HardwareAccelerationError):
    """Raised when accelerated execution provider fails during session initialization or inference."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_ACCELERATED_PROVIDER_FAILED")


class InsufficientMemoryError(SnapShieldError):
    """Raised when system memory is insufficient for large document processing or model inference."""
    def __init__(self, message: str = "Insufficient system memory to complete operation.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_INSUFFICIENT_MEMORY")


# =========================================================================
# Policy & Redaction Exceptions
# =========================================================================

class PolicyViolationError(SnapShieldError):
    """Raised when an operation attempts to bypass an active blocking policy."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_POLICY_VIOLATION")


class PolicyExecutionError(SnapShieldError):
    """Raised when policy evaluation fails or encounters misconfiguration."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_POLICY_EXECUTION")


class SecurityBoundaryError(SnapShieldError):
    """Critical exception raised if raw unredacted data is directed toward an external service."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_SECURITY_BOUNDARY_VIOLATION")


class RedactionError(SnapShieldError):
    """Raised when text sanitization or masking fails to produce safe output."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_REDACTION_FAILED")


# =========================================================================
# External LLM & Network Exceptions
# =========================================================================

class LLMServiceError(SnapShieldError):
    """Raised when external LLM communication or response processing fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: str = "ERR_LLM_SERVICE") -> None:
        super().__init__(message, details, error_code)


class NetworkError(LLMServiceError):
    """Raised when external network connectivity or DNS resolution fails."""
    def __init__(self, message: str = "Network connection to external AI service failed or timed out.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_NETWORK_DISCONNECTED")


class AuthenticationError(LLMServiceError):
    """Raised when an external service rejects authentication credentials (e.g. invalid API key)."""
    def __init__(self, message: str = "External AI authentication failed: Invalid or expired API key.", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details, error_code="ERR_AUTHENTICATION_FAILED")


# =========================================================================
# User-Friendly Error Diagnostic Schema
# =========================================================================

@dataclass
class ErrorDiagnostic:
    """Structured diagnostic payload for clean UI error rendering without raw stack traces."""
    title: str
    user_message: str
    recovery_action: str
    degradation_status: str
    error_code: str
    is_recoverable: bool = True
    technical_details: Dict[str, Any] = field(default_factory=dict)
    traceback_str: Optional[str] = None


def classify_error(exc: Exception) -> ErrorDiagnostic:
    """Classifies any system exception into a user-friendly, actionable diagnostic."""
    tb = traceback.format_exc()

    # 1. Empty File / Input
    if isinstance(exc, EmptyFileError):
        return ErrorDiagnostic(
            title="Empty Input Detected",
            user_message="The provided text or document has 0 characters. Please type text or upload a document.",
            recovery_action="Select a sample document from 'Load Demo Document' or paste content above.",
            degradation_status="Checkpoint Idle • Awaiting input payload",
            error_code=exc.error_code,
            technical_details=exc.details,
            traceback_str=tb,
        )

    # 2. Oversized File
    if isinstance(exc, OversizedFileError):
        return ErrorDiagnostic(
            title="File Size Limit Exceeded",
            user_message=exc.message,
            recovery_action="Please upload a file under 25 MB or an image under 25 Megapixels.",
            degradation_status="Checkpoint Aborted • Payload rejected for memory protection",
            error_code=exc.error_code,
            technical_details=exc.details,
            traceback_str=tb,
        )

    # 3. Invalid File Format
    if isinstance(exc, InvalidFileFormatError) or isinstance(exc, InputValidationError):
        return ErrorDiagnostic(
            title="Unsupported File Format",
            user_message=str(exc),
            recovery_action="SnapShield supports .txt, .pdf, .png, .jpg, and .jpeg files.",
            degradation_status="Checkpoint Aborted • Ingestion format mismatch",
            error_code=getattr(exc, "error_code", "ERR_INPUT_VALIDATION"),
            technical_details=getattr(exc, "details", {}),
            traceback_str=tb,
        )

    # 4. PDF Extraction Failure
    if isinstance(exc, (PDFExtractionError, ExtractionError)):
        return ErrorDiagnostic(
            title="PDF Extraction Failed",
            user_message="SnapShield could not extract text from this PDF. It may be encrypted, scanned, or corrupted.",
            recovery_action="For scanned documents, please convert pages to PNG/JPG images for Local OCR scanning.",
            degradation_status="Extraction Aborted • PDF parser encountered invalid byte stream",
            error_code=getattr(exc, "error_code", "ERR_PDF_EXTRACTION_FAILED"),
            technical_details=getattr(exc, "details", {}),
            traceback_str=tb,
        )

    # 5. OCR Failure
    if isinstance(exc, OCRError):
        return ErrorDiagnostic(
            title="Local OCR Processing Notice",
            user_message=str(exc),
            recovery_action="Verify that local OCR is installed or paste raw text directly into the prompt box.",
            degradation_status="OCR Fallback • Local image text extraction unavailable",
            error_code=getattr(exc, "error_code", "ERR_OCR_FAILED"),
            technical_details=getattr(exc, "details", {}),
            traceback_str=tb,
        )

    # 6. AI Model Unavailable
    if isinstance(exc, ModelUnavailableError):
        return ErrorDiagnostic(
            title="Local AI Model Notice",
            user_message="Local ONNX token classification model was not found on-device.",
            recovery_action="Deterministic regex and checksum detectors continue scanning seamlessly.",
            degradation_status="✓ Deterministic Scanning Active (Rule-based detection continues uninterrupted)",
            error_code=exc.error_code,
            technical_details=exc.details,
            traceback_str=tb,
        )

    # 7. Hardware Acceleration Failure
    if isinstance(exc, HardwareAccelerationError):
        return ErrorDiagnostic(
            title="Hardware Accelerator Notice",
            user_message="Hardware acceleration (QNN / DirectML) could not be engaged on this device.",
            recovery_action="System automatically transitioned to verified CPU inference.",
            degradation_status="✓ CPU Fallback Active (Inference continues safely on CPU)",
            error_code=getattr(exc, "error_code", "ERR_HARDWARE_ACCELERATION"),
            technical_details=getattr(exc, "details", {}),
            traceback_str=tb,
        )

    # 8. Memory Limitation
    if isinstance(exc, (InsufficientMemoryError, MemoryError)):
        return ErrorDiagnostic(
            title="Insufficient System Memory",
            user_message="Available system memory was exceeded during document processing or inference.",
            recovery_action="Try analyzing a smaller section of text or closing background applications.",
            degradation_status="Checkpoint Paused • Memory threshold reached",
            error_code="ERR_INSUFFICIENT_MEMORY",
            technical_details={"error": str(exc)},
            traceback_str=tb,
        )

    # 9. Authentication Error (LLM)
    if isinstance(exc, AuthenticationError):
        return ErrorDiagnostic(
            title="External AI Authentication Failed",
            user_message="The external AI provider rejected the API key (401/403 Unauthorized).",
            recovery_action="Verify LLM_API_KEY in your .env configuration. Your sanitized document remains safe locally.",
            degradation_status="✓ Local Sanitized Output Preserved (Air-gapped local privacy intact)",
            error_code=exc.error_code,
            technical_details=exc.details,
            traceback_str=tb,
        )

    # 10. Network Error (LLM)
    if isinstance(exc, NetworkError):
        return ErrorDiagnostic(
            title="Network Connection Notice",
            user_message="Unable to reach external AI provider endpoint (Network disconnected or timed out).",
            recovery_action="Check your internet connection. Sanitized document is ready for local copy/download.",
            degradation_status="✓ Local Sanitized Output Preserved (Zero egress occurred)",
            error_code=exc.error_code,
            technical_details=exc.details,
            traceback_str=tb,
        )

    # 11. Security Boundary Violation
    if isinstance(exc, SecurityBoundaryError):
        return ErrorDiagnostic(
            title="Security Boundary Intercepted",
            user_message="SnapShield blocked an unsafe outbound dispatch attempt.",
            recovery_action="Ensure all review items are approved and document is sanitized before sending.",
            degradation_status="🛡️ Air-Gap Perimeter Enforced • Egress strictly halted",
            error_code=exc.error_code,
            technical_details=exc.details,
            traceback_str=tb,
        )

    # 12. Generic / Uncaught Error
    return ErrorDiagnostic(
        title="Processing Notice",
        user_message=f"An operational notice occurred during checkpoint processing: {str(exc)}",
        recovery_action="Review document input or inspect technical diagnostics in developer debug mode.",
        degradation_status="Checkpoint Recoverable • Core privacy safeguards remain active",
        error_code=getattr(exc, "error_code", "ERR_UNEXPECTED"),
        technical_details={"exception_type": type(exc).__name__, "message": str(exc)},
        traceback_str=tb,
    )
