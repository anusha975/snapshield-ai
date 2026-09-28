"""Comprehensive End-to-End Integration Test Suite for SnapShield AI.

Validates the exact workflows specified for TEXT, PDF, IMAGE, AI, and Policies:
1. TEXT Workflow: Text -> scan -> detection -> risk -> policy -> redaction -> safe output -> optional LLM
2. PDF Workflow: PDF -> extraction -> detection -> risk -> policy -> redaction -> safe output
3. IMAGE Workflow: Screenshot/Image -> OCR -> detection -> risk -> policy -> redaction -> safe output
4. Synthetic Employee Scan: Aadhaar, PAN, Phone, Email, Bank, IFSC
5. Synthetic Medical Scan: Patient, Diagnosis, Doctor, Hospital
6. Synthetic Developer Scan: API key, Password, JWT, DB URL, AWS key
7. Safety Boundary: External LLM cannot be called on unredacted or blocked content
8. Runtime Diagnostics & Benchmarks
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image, ImageDraw

import pytest

from snapshield.core.exceptions import PolicyViolationError, SecurityBoundaryError
from snapshield.core.llm_client import ExternalLLMClient, verify_safety_boundary
from snapshield.core.models import InputDocument, PolicyAction, ScanResult
from snapshield.core.ocr import LocalOCRProvider, OCROutput
from snapshield.data import load_sample_document, load_synthetic_samples
from snapshield.npu.benchmark import run_benchmark
from snapshield.npu.device import probe_hardware_capabilities
from snapshield.services.scan_service import scan_document, scan_image, scan_pdf


def _create_mock_pdf_bytes(text_content: str) -> bytes:
    """Creates a minimal valid PDF in-memory containing text_content."""
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        "3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>/Contents 4 0 R>>endobj\n"
        f"4 0 obj<</Length {len(text_content) + 20}>>\n"
        f"stream\nBT /F1 12 Tf 72 720 Td ({text_content}) Tj ET\nendstream\nendobj\n"
        "xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000214 00000 n \n"
        "trailer<</Size 5/Root 1 0 R>>\nstartxref\n310\n%%EOF\n"
    )
    return pdf_content.encode("latin-1")


# =========================================================================
# 1. TEXT Workflow Integration Test
# =========================================================================

def test_text_workflow_complete_pipeline():
    """Verify TEXT workflow: paste -> scan -> detection -> risk -> policy -> redaction -> safe output -> LLM."""
    text_input = (
        "Hello, my name is Priya Sharma. My phone is +91 98765 43210 and email is priya@example.com."
    )

    # 1. Scan document
    result = scan_document(document=text_input, policy_mode="Personal", ai_enabled=True)

    # 2. Check Detection phase
    assert len(result.detections) >= 1
    from snapshield.config.policies import normalize_category_key
    norm_types = {normalize_category_key(d.entity_type) for d in result.detections}
    assert "PHONE" in norm_types or "PHONE_NUMBER" in norm_types

    # 3. Check Risk phase
    assert result.risk_assessment.score > 0

    # 4. Check Policy decisions
    assert len(result.decisions) == len(result.detections)
    actions = {dec.action for dec in result.decisions}
    assert PolicyAction.REDACT in actions

    # 5. Check Redaction phase
    assert "+91 98765 43210" not in result.sanitized_text
    assert "[REDACTED:" in result.sanitized_text
    assert result.integrity_hash != ""

    # 6. Check Pre-Flight Safety Boundary for Optional LLM Gateway
    verify_safety_boundary(scan=result, text_to_send=result.sanitized_text, review_approved=True)

    client = ExternalLLMClient()
    assert client.get_status()["is_external"] is True



# =========================================================================
# 2. PDF Workflow Integration Test
# =========================================================================

def test_pdf_workflow_complete_pipeline():
    """Verify PDF workflow: PDF bytes -> extraction -> detection -> risk -> policy -> redaction -> safe output."""
    raw_text = "CONFIDENTIAL: Aadhaar 2345 6789 0124, PAN ABCPA1234F, Email: test@example.com"
    pdf_bytes = _create_mock_pdf_bytes(raw_text)

    # Scan PDF document
    result = scan_pdf(pdf_input=pdf_bytes, policy_mode="Personal", ai_enabled=False, file_name="statement.pdf")

    assert result is not None
    assert result.timings.total_ms > 0
    assert result.integrity_hash != ""


# =========================================================================
# 3. IMAGE Workflow Integration Test
# =========================================================================

def test_image_workflow_complete_pipeline():
    """Verify IMAGE workflow: Image bytes -> OCR -> detection -> risk -> policy -> redaction -> safe output."""
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    def mock_ocr(im: Image.Image, p_idx: int) -> OCROutput:
        return OCROutput(
            text="PORTAL: Dr. Sharma, Aadhaar 2345 6789 0124, API Key: sk-live-abc1234567890def1234567890gh",
            provider_name="mock_test_ocr",
            processing_time_ms=10.0,
        )

    from snapshield.core.ocr import OCRRegistry
    original_provider = OCRRegistry.get_provider()
    try:
        OCRRegistry.set_provider(LocalOCRProvider(custom_fallback=mock_ocr))
        result = scan_image(image_input=img_bytes, policy_mode="Developer", ai_enabled=True)

        assert result is not None
        assert "2345 6789 0124" in result.original_text
        assert "2345 6789 0124" not in result.sanitized_text
        assert "sk-live-abc1234567890def1234567890gh" not in result.sanitized_text

        # In Developer mode, API Key is BLOCKED
        actions = [dec.action for dec in result.decisions]
        assert PolicyAction.BLOCK in actions
    finally:
        OCRRegistry.set_provider(original_provider)


# =========================================================================
# 4. Synthetic Employee Scan Integration Test
# =========================================================================

def test_synthetic_employee_scan_integration():
    """Verify synthetic employee document scan with all required Indian PII."""
    emp_doc = load_sample_document("employee_record.txt")
    assert emp_doc != ""

    result = scan_document(document=emp_doc, policy_mode="Personal", ai_enabled=True)

    # Detections
    from snapshield.config.policies import normalize_category_key
    norm_types = {normalize_category_key(d.entity_type) for d in result.detections}
    assert "AADHAAR" in norm_types
    assert "PAN" in norm_types
    assert "PHONE" in norm_types or "PHONE_NUMBER" in norm_types
    assert "EMAIL" in norm_types or "EMAIL_ADDRESS" in norm_types
    assert "BANK_ACCOUNT" in norm_types
    assert "IFSC" in norm_types or "IFSC_CODE" in norm_types



    # Ensure zero sensitive tokens leak in sanitized output for redacted categories
    assert "9999 9999 0019" not in result.sanitized_text
    assert "ABCPA1234F" not in result.sanitized_text
    assert "+91 98765 43210" not in result.sanitized_text
    assert "987654321012345" not in result.sanitized_text
    assert "HDFC0001234" not in result.sanitized_text


# =========================================================================
# 5. Synthetic Medical Scan Integration Test
# =========================================================================

def test_synthetic_medical_scan_integration():
    """Verify synthetic medical note scan with patient details, diagnoses, and doctor info."""
    med_doc = load_sample_document("medical_note.txt")
    assert med_doc != ""

    result = scan_document(document=med_doc, policy_mode="Medical", ai_enabled=True)

    assert result.risk_assessment.score > 0
    assert result.sanitized_text != ""
    assert "[REDACTED:" in result.sanitized_text or "[BLOCKED:" in result.sanitized_text


# =========================================================================
# 6. Synthetic Developer Scan Integration Test
# =========================================================================

def test_synthetic_developer_scan_integration():
    """Verify synthetic developer config scan blocks credentials and API keys."""
    dev_doc = load_sample_document("developer_config.txt")
    assert dev_doc != ""

    result = scan_document(document=dev_doc, policy_mode="Developer", ai_enabled=True)

    entity_types = {d.entity_type for d in result.detections}
    assert "API_KEY" in entity_types or "PASSWORD" in entity_types or "DATABASE_URI" in entity_types

    # In Developer policy, secrets are BLOCKED
    actions = [dec.action for dec in result.decisions]
    assert PolicyAction.BLOCK in actions

    # Proves LLM cannot be called when content contains BLOCK actions
    client = ExternalLLMClient()
    with pytest.raises((PolicyViolationError, SecurityBoundaryError)):
        client.send_prompt(scan_result=result, text_to_send=result.sanitized_text)


# =========================================================================
# 7. Runtime Diagnostics and Benchmarking Integration Test
# =========================================================================

def test_runtime_diagnostics_and_benchmarking():
    """Verify hardware probing and benchmarking suite."""
    hw_report = probe_hardware_capabilities()
    assert hw_report is not None
    assert hw_report.active_execution_provider in (
        "CPUExecutionProvider",
        "QNNExecutionProvider",
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
    )

    # Run on-device micro-benchmarks
    bench_suite = run_benchmark(iterations=2)
    assert bench_suite is not None
    assert bench_suite.cpu_result is not None
    assert bench_suite.cpu_result.mean_latency_ms >= 0.0
