"""SnapShield AI Core Module.

Contains domain types, exception hierarchies, and privacy-safe logging.
"""

from .types import (
    CheckpointResult,
    DetectionSource,
    DocumentType,
    EntityCategory,
    Finding,
    PolicyAction,
    PolicyDecision,
    RedactionResult,
    RiskAssessment,
    RiskLevel,
)
from .models import (
    BenchmarkResult,
    Detection,
    InputDocument,
    OCRResult,
    PolicyDecision,
    RiskAssessment,
    RuntimeStatus,
    ScanResult,
    ScanTiming,
    SeverityLevel,
    mask_sensitive_value,
)
from .exceptions import (
    DetectionError,
    ExtractionError,
    HardwareAccelerationError,
    InputValidationError,
    OCRError,
    PolicyExecutionError,
    PolicyViolationError,
    RedactionError,
    SecurityBoundaryError,
    SnapShieldError,
)
from .ai_detector import (
    AIDetector,
    LocalAIDetector,
    ModelMetadata,
    PreprocessedInput,
    RawModelOutput,
    TARGET_CATEGORIES,
)
from .code_detectors import detect_credentials
from .detectors import detect_structured
from .result_merger import ResultMerger, merge_detections, scan_text
from .risk_engine import (
    RiskContributor,
    RiskEngine,
    RiskEngineConfig,
    RiskReport,
    calculate_risk,
)
from .policy import (
    PolicyEngine,
    SUPPORTED_MODES,
    apply_policy,
)
from .redactor import (
    Redactor,
    redact,
    redact_text,
)
from .extractor import extract_pdf
from .input_handler import (
    FileValidationResult,
    create_document_from_upload,
    create_text_document,
    validate_uploaded_file,
)
from .visualization import annotate_image_with_detections
from .ocr import (
    AcceleratedOCRProvider,
    LocalOCRProvider,
    OCRBoundingBox,
    OCROutput,
    OCRPageOutput,
    OCRProvider,
    OCRRegistry,
    get_ocr_provider,
)
from .verhoeff import (
    compute_verhoeff_check_digit,
    generate_synthetic_aadhaar,
    is_valid_aadhaar,
    is_valid_verhoeff,
)
from .llm_client import (
    ExternalLLMClient,
    LLMClient,
    LLMClientError,
    LLMConfig,
    LLMResponse,
    get_llm_client,
    verify_safety_boundary,
)
from .logger import logger, setup_logger

__all__ = [
    "AIDetector",
    "AcceleratedOCRProvider",
    "BenchmarkResult",
    "CheckpointResult",
    "Detection",
    "annotate_image_with_detections",
    "DetectionError",
    "DetectionSource",
    "DocumentType",
    "EntityCategory",
    "ExternalLLMClient",
    "ExtractionError",
    "FileValidationResult",
    "Finding",
    "HardwareAccelerationError",
    "InputDocument",
    "InputValidationError",
    "LLMClient",
    "LLMClientError",
    "LLMConfig",
    "LLMResponse",
    "LocalAIDetector",
    "LocalOCRProvider",
    "ModelMetadata",
    "PreprocessedInput",
    "RawModelOutput",
    "OCRBoundingBox",
    "OCRError",
    "OCROutput",
    "OCRPageOutput",
    "OCRProvider",
    "OCRRegistry",
    "OCRResult",
    "PolicyAction",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyExecutionError",
    "PolicyViolationError",
    "RedactionError",
    "RedactionResult",
    "Redactor",
    "ResultMerger",
    "RiskAssessment",
    "RiskContributor",
    "RiskEngine",
    "RiskEngineConfig",
    "RiskLevel",
    "RiskReport",
    "RuntimeStatus",
    "ScanResult",
    "ScanTiming",
    "SecurityBoundaryError",
    "SeverityLevel",
    "SnapShieldError",
    "SUPPORTED_MODES",
    "apply_policy",
    "calculate_risk",
    "compute_verhoeff_check_digit",
    "create_document_from_upload",
    "create_text_document",
    "detect_credentials",
    "detect_structured",
    "extract_pdf",
    "generate_synthetic_aadhaar",
    "get_llm_client",
    "get_ocr_provider",
    "is_valid_aadhaar",
    "is_valid_verhoeff",
    "logger",
    "mask_sensitive_value",
    "merge_detections",
    "redact",
    "redact_text",
    "scan_text",
    "setup_logger",
    "validate_uploaded_file",
    "verify_safety_boundary",
]
