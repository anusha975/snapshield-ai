"""Unified document scanning service for SnapShield AI.

Implements the end-to-end processing pipeline:
INPUT
→ extraction
→ OCR (if required)
→ structured detection
→ secret detection
→ AI detection
→ result merger
→ risk engine
→ policy
→ redaction
→ ScanResult

Architectural Guarantees:
- Completely decoupled from UI presentation code (no Streamlit imports).
- Accurately measures individual phase timings using time.perf_counter():
  - extraction_ms
  - ocr_ms
  - structured_detection_ms
  - ai_inference_ms
  - risk_ms
  - policy_ms
  - redaction_ms
  - total_ms
- Graceful degradation: If AI is unavailable, continues seamlessly using deterministic
  detection and clearly exposes the AI unavailable state.
- Strictly enforces privacy invariants: decoupling original input from sanitized output.
"""

from __future__ import annotations

import io
import time
from typing import Any, List, Optional, Union

from snapshield.core.ai_detector import LocalAIDetector
from snapshield.core.code_detectors import detect_credentials
from snapshield.core.detectors import detect_structured
from snapshield.core.extractor import extract_pdf
from snapshield.core.input_handler import create_text_document
from snapshield.core.logger import logger
from snapshield.core.models import (
    Detection,
    InputDocument,
    PolicyAction,
    PolicyDecision,
    RiskAssessment,
    RuntimeStatus,
    ScanResult,
    ScanTiming,
)
from snapshield.core.ocr import get_ocr_provider
from snapshield.core.policy import PolicyEngine
from snapshield.core.redactor import Redactor
from snapshield.core.result_merger import get_default_ai_detector, merge_detections
from snapshield.core.risk_engine import RiskEngine
from snapshield.npu.device import probe_hardware_capabilities


class ScanService:
    """Orchestrates the complete SnapShield AI privacy checkpoint pipeline."""

    def __init__(self, default_policy_mode: str = "Personal") -> None:
        self.default_policy_mode = default_policy_mode

    def scan(
        self,
        document: Union[str, InputDocument, Any],
        policy_mode: Optional[str] = None,
        ai_enabled: bool = True,
    ) -> ScanResult:
        """Executes full scan on document using the configured policy mode."""
        return scan_document(
            document=document,
            policy_mode=policy_mode or self.default_policy_mode,
            ai_enabled=ai_enabled,
        )

    def scan_image(
        self,
        image_input: Union[bytes, Any],
        policy_mode: Optional[str] = None,
        ai_enabled: bool = True,
        file_name: str = "screenshot.png",
    ) -> ScanResult:
        """Scans image input through local OCR and privacy checkpoint pipeline."""
        return scan_image(
            image_input=image_input,
            policy_mode=policy_mode or self.default_policy_mode,
            ai_enabled=ai_enabled,
            file_name=file_name,
        )

    def scan_pdf(
        self,
        pdf_input: Union[bytes, Any],
        policy_mode: Optional[str] = None,
        ai_enabled: bool = True,
        file_name: str = "document.pdf",
    ) -> ScanResult:
        """Scans PDF document through local extraction and privacy checkpoint pipeline."""
        return scan_pdf(
            pdf_input=pdf_input,
            policy_mode=policy_mode or self.default_policy_mode,
            ai_enabled=ai_enabled,
            file_name=file_name,
        )



def scan_document(
    document: Union[str, InputDocument, Any],
    policy_mode: str = "Personal",
    ai_enabled: bool = True,
) -> ScanResult:
    """Executes the complete SnapShield AI privacy checkpoint pipeline.

    Args:
        document: Raw text string or InputDocument instance.
        policy_mode: Policy profile ('Personal', 'Developer', 'Medical'). Defaults to 'Personal'.
        ai_enabled: Whether to run the local AI contextual detector. Defaults to True.

    Returns:
        ScanResult containing sanitized text, detections, decisions, risk assessment,
        phase telemetry timings, hardware runtime status, and integrity hashes.
    """
    t_global_start = time.perf_counter()
    errors: List[str] = []
    ocr_boxes: List[Any] = []

    # Initialize individual phase timings
    extraction_ms = 0.0
    ocr_ms = 0.0
    structured_detection_ms = 0.0
    ai_inference_ms = 0.0
    risk_ms = 0.0
    policy_ms = 0.0
    redaction_ms = 0.0

    # =================================================================
    # Stage 1: Extraction & Ingestion
    # =================================================================
    original_text = ""
    doc_type = "RAW_TEXT"

    t_ext_start = time.perf_counter()

    if isinstance(document, str):
        original_text = document
        doc_type = "RAW_TEXT"
        extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0

    elif isinstance(document, InputDocument):
        doc_type = document.document_type

        if document.document_type == "RAW_TEXT":
            original_text = document.raw_content or ""
            extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0

        elif document.document_type == "PDF":
            if document.raw_bytes:
                try:
                    pdf_result = extract_pdf(
                        file_bytes=document.raw_bytes,
                        file_name=document.file_name,
                    )
                    original_text = pdf_result.raw_content or ""
                    extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0

                    # Check if PDF is scanned or requires OCR
                    if pdf_result.metadata.get("ocr_required") or not original_text.strip():
                        t_ocr_start = time.perf_counter()
                        ocr_provider = get_ocr_provider()
                        try:
                            if hasattr(ocr_provider, "extract_from_image"):
                                ocr_out = ocr_provider.extract_from_image(document.raw_bytes)
                            else:
                                ocr_out = ocr_provider.extract_text(document.raw_bytes)
                            if ocr_out.text.strip():
                                original_text = ocr_out.text
                                ocr_boxes = getattr(ocr_out, "bounding_boxes", [])
                        except Exception as ocr_exc:
                            errors.append(f"OCR fallback failed: {ocr_exc}")
                        ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

                except Exception as exc:
                    errors.append(f"PDF extraction error: {exc}")
                    original_text = document.raw_content or ""
                    extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0
            else:
                original_text = document.raw_content or ""
                extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0

        elif document.document_type == "IMAGE":
            extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0
            t_ocr_start = time.perf_counter()
            if document.raw_bytes:
                try:
                    ocr_provider = get_ocr_provider()
                    if hasattr(ocr_provider, "extract_from_image"):
                        ocr_out = ocr_provider.extract_from_image(document.raw_bytes)
                    else:
                        ocr_out = ocr_provider.extract_text(document.raw_bytes)
                    original_text = ocr_out.text
                    ocr_boxes = getattr(ocr_out, "bounding_boxes", [])
                except Exception as ocr_exc:
                    errors.append(f"Image OCR failed: {ocr_exc}")
                    original_text = document.raw_content or ""
            else:
                original_text = document.raw_content or ""
            ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

        else:
            original_text = document.raw_content or ""
            extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0

    else:
        # Fallback string representation
        original_text = str(document)
        extraction_ms = (time.perf_counter() - t_ext_start) * 1000.0

    # =================================================================
    # Stage 2: Structured & Secret Detection
    # =================================================================
    t_det_start = time.perf_counter()
    struct_findings: List[Detection] = []
    cred_findings: List[Detection] = []

    try:
        struct_findings = detect_structured(original_text)
    except Exception as exc:
        errors.append(f"Structured detector error: {exc}")

    try:
        cred_findings = detect_credentials(original_text)
    except Exception as exc:
        errors.append(f"Secret detector error: {exc}")

    structured_detection_ms = (time.perf_counter() - t_det_start) * 1000.0

    # =================================================================
    # Stage 3: Local AI Detection & Status Exposure
    # =================================================================
    ai_findings: List[Detection] = []
    ai_status = "disabled"

    if ai_enabled:
        t_ai_start = time.perf_counter()
        try:
            ai_detector = get_default_ai_detector()
            if ai_detector.is_available:
                ai_findings = ai_detector.detect(original_text)
                ai_status = "active_fallback" if ai_detector.metadata.is_fallback else "active_onnx"
                if ai_detector.metadata.is_fallback:
                    errors.append("Local AI running in development fallback mode (ONNX model uninstalled).")
            else:
                ai_status = "unavailable"
                errors.append("Local AI detector is unavailable; continued using deterministic detection.")
        except Exception as exc:
            ai_status = "failed"
            errors.append(f"Local AI detector error ({exc}); continued using deterministic detection.")

        ai_inference_ms = (time.perf_counter() - t_ai_start) * 1000.0
    else:
        ai_status = "disabled"

    # =================================================================
    # Stage 4: Result Merger
    # =================================================================
    all_raw_detections = struct_findings + cred_findings + ai_findings
    merged_detections = merge_detections(all_raw_detections, text=original_text)

    # =================================================================
    # Stage 5: Risk Engine Evaluation
    # =================================================================
    t_risk_start = time.perf_counter()
    try:
        risk_engine = RiskEngine()
        risk_report = risk_engine.evaluate(merged_detections)
        risk_assessment = risk_report.to_risk_assessment()
    except Exception as exc:
        errors.append(f"Risk evaluation error: {exc}")
        risk_assessment = RiskAssessment(score=0.0, findings_count=len(merged_detections))
    risk_ms = (time.perf_counter() - t_risk_start) * 1000.0

    # =================================================================
    # Stage 6: Policy Enforcement
    # =================================================================
    t_pol_start = time.perf_counter()
    try:
        policy_engine = PolicyEngine(mode=policy_mode)
        decisions = policy_engine.apply_policy(merged_detections)
    except Exception as exc:
        errors.append(f"Policy enforcement error: {exc}")
        decisions = []
    policy_ms = (time.perf_counter() - t_pol_start) * 1000.0

    # =================================================================
    # Stage 7: Redaction & Sanitization
    # =================================================================
    t_red_start = time.perf_counter()
    try:
        redaction_res = Redactor.redact(
            original_text=original_text,
            detections=merged_detections,
            decisions=decisions,
            mode=policy_mode,
        )
        sanitized_text = redaction_res.sanitized_text
        integrity_hash = redaction_res.integrity_hash
    except Exception as exc:
        errors.append(f"Redaction error: {exc}")
        sanitized_text = "[REDACTION FAILED]"
        integrity_hash = ""
    redaction_ms = (time.perf_counter() - t_red_start) * 1000.0

    # =================================================================
    # Stage 8: Visualization Overlay (if OCR bounding boxes are present)
    # =================================================================
    annotated_bytes: Optional[bytes] = None
    annotated_count: int = 0
    if isinstance(document, InputDocument) and document.document_type == "IMAGE" and document.raw_bytes and ocr_boxes:
        try:
            from snapshield.core.visualization import annotate_image_with_detections
            annotated_bytes, annotated_count = annotate_image_with_detections(
                image_input=document.raw_bytes,
                extracted_text=original_text,
                bounding_boxes=ocr_boxes,
                detections=merged_detections,
                decisions=decisions,
            )
        except Exception as viz_exc:
            errors.append(f"Image visualization overlay failed: {viz_exc}")

    # =================================================================
    # Stage 9: Hardware Telemetry & Runtime Status
    # =================================================================
    try:
        hw_report = probe_hardware_capabilities()
        device_backend = hw_report.acceleration_summary
        if ai_status == "unavailable":
            device_backend += " (AI Detector Unavailable)"
        elif ai_status == "active_fallback":
            device_backend += " (AI Dev Fallback Active)"

        runtime_status = RuntimeStatus(
            os_name=hw_report.os_name,
            machine_arch=hw_report.machine_arch,
            processor=hw_report.processor_name,
            active_execution_provider=hw_report.active_execution_provider,
            is_accelerated=hw_report.is_accelerated,
            device_backend=device_backend,
        )
    except Exception as exc:
        errors.append(f"Hardware probe error: {exc}")
        runtime_status = RuntimeStatus(
            os_name="Unknown",
            machine_arch="Unknown",
            processor="Unknown",
            active_execution_provider="CPUExecutionProvider",
            is_accelerated=False,
            device_backend="Verified CPU Fallback",
        )

    # =================================================================
    # Stage 10: Assembly of ScanResult
    # =================================================================
    t_global_end = time.perf_counter()
    total_ms = (t_global_end - t_global_start) * 1000.0

    detection_total_ms = structured_detection_ms + ai_inference_ms

    timings = ScanTiming(
        extraction_ms=round(extraction_ms, 2),
        ocr_ms=round(ocr_ms, 2),
        structured_detection_ms=round(structured_detection_ms, 2),
        ai_inference_ms=round(ai_inference_ms, 2),
        detection_ms=round(detection_total_ms, 2),
        risk_ms=round(risk_ms, 2),
        risk_assessment_ms=round(risk_ms, 2),
        policy_ms=round(policy_ms, 2),
        redaction_ms=round(redaction_ms, 2),
        total_ms=round(total_ms, 2),
    )

    logger.info(
        "Scan complete for '%s' (type=%s, mode=%s, ai=%s): Detections=%d, TotalTime=%.2fms",
        getattr(document, "file_name", "TextPayload") or "TextPayload",
        doc_type,
        policy_mode,
        ai_status,
        len(merged_detections),
        total_ms,
    )

    return ScanResult(
        original_text=original_text,
        sanitized_text=sanitized_text,
        detections=merged_detections,
        decisions=decisions,
        risk_assessment=risk_assessment,
        timings=timings,
        runtime_status=runtime_status,
        errors=errors,
        integrity_hash=integrity_hash,
        ocr_bounding_boxes=[(b.model_dump() if hasattr(b, "model_dump") else b) for b in ocr_boxes],
        annotated_image_bytes=annotated_bytes,
        annotated_regions_count=annotated_count,
    )


def scan_image(
    image_input: Union[bytes, Any],
    policy_mode: str = "Personal",
    ai_enabled: bool = True,
    file_name: str = "screenshot.png",
) -> ScanResult:
    """Scans an image or screenshot through local OCR and privacy checkpoint pipeline.

    Workflow:
    Image -> Local OCR -> extracted text -> hybrid detection -> risk -> policy -> redaction

    - Reuses the existing detection pipeline.
    - Does not duplicate detection code.
    - If OCR provides bounding boxes, creates a visualization layer that highlights detected regions.
    - Supports PNG, JPG, JPEG.
    - Handles OCR failures cleanly.
    - Never sends images to an external service for detection.
    """
    raw_bytes: bytes = b""
    detected_format: Optional[str] = None
    validation_error: Optional[str] = None

    try:
        from PIL import Image

        if isinstance(image_input, bytes):
            raw_bytes = image_input
            bio = io.BytesIO(raw_bytes)
            with Image.open(bio) as img:
                detected_format = (img.format or "").upper()
        elif isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                validation_error = f"Image file not found at: {p}"
            else:
                raw_bytes = p.read_bytes()
                with Image.open(p) as img:
                    detected_format = (img.format or "").upper()
        elif hasattr(image_input, "read"):
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            raw_bytes = image_input.read()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                detected_format = (img.format or "").upper()
        elif hasattr(image_input, "save") and hasattr(image_input, "format"):
            # PIL Image instance
            detected_format = (image_input.format or "PNG").upper()
            buf = io.BytesIO()
            save_fmt = "JPEG" if detected_format in ("JPEG", "JPG") else "PNG"
            image_input.save(buf, format=save_fmt)
            raw_bytes = buf.getvalue()
        else:
            validation_error = f"Unsupported image input type: {type(image_input).__name__}"
    except Exception as exc:
        validation_error = f"Failed to load image: Corrupted or invalid image data ({exc})"

    # Validate format: PNG, JPG, JPEG
    if not validation_error and detected_format:
        if detected_format not in ("PNG", "JPG", "JPEG"):
            validation_error = (
                f"Unsupported image format '{detected_format}'. "
                "Supported formats are PNG, JPG, JPEG."
            )

    if validation_error:
        timing = ScanTiming(total_ms=0.0)
        risk = RiskAssessment(score=0.0, findings_count=0)
        runtime = RuntimeStatus(
            os_name="Local",
            machine_arch="Local",
            processor="Local",
            active_execution_provider="CPUExecutionProvider",
            is_accelerated=False,
            device_backend="Local Checkpoint",
        )
        return ScanResult(
            original_text="",
            sanitized_text="",
            detections=[],
            decisions=[],
            risk_assessment=risk,
            timings=timing,
            runtime_status=runtime,
            errors=[validation_error],
        )

    doc = InputDocument(
        document_type="IMAGE",
        raw_bytes=raw_bytes,
        file_name=file_name,
    )
    return scan_document(
        document=doc,
        policy_mode=policy_mode,
        ai_enabled=ai_enabled,
    )


def scan_pdf(
    pdf_input: Union[bytes, str, Path, Any],
    policy_mode: str = "Personal",
    ai_enabled: bool = True,
    file_name: str = "document.pdf",
) -> ScanResult:
    """Scans a PDF document through local extraction and privacy checkpoint pipeline.

    Workflow:
    PDF -> Text Extraction / OCR fallback -> detection -> risk -> policy -> redaction -> safe output
    """
    raw_bytes: bytes = b""
    if isinstance(pdf_input, bytes):
        raw_bytes = pdf_input
    elif isinstance(pdf_input, (str, Path)):
        p = Path(pdf_input)
        if p.exists():
            raw_bytes = p.read_bytes()
            file_name = p.name
    elif hasattr(pdf_input, "read"):
        if hasattr(pdf_input, "seek"):
            pdf_input.seek(0)
        raw_bytes = pdf_input.read()
        if hasattr(pdf_input, "name"):
            file_name = pdf_input.name

    doc = InputDocument(
        document_type="PDF",
        raw_bytes=raw_bytes,
        file_name=file_name,
    )
    return scan_document(
        document=doc,
        policy_mode=policy_mode,
        ai_enabled=ai_enabled,
    )

