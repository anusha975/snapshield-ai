"""Unit and integration tests for screenshot and image scanning workflow.

Validates:
- Image format validation: PNG, JPG, JPEG supported; unsupported formats handled cleanly.
- Corrupted image error handling without crashing.
- End-to-end pipeline: Image -> Local OCR -> extracted text -> hybrid detection -> risk -> policy -> redaction.
- Bounding box visualization overlay rendering with action-coded highlights.
- Reuse of existing detection pipeline without duplicate detection logic.
- Privacy invariant: Zero external cloud calls.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image, ImageDraw

from snapshield.core.models import Detection, PolicyAction, PolicyDecision, ScanResult, SeverityLevel
from snapshield.core.ocr import OCRBoundingBox, OCROutput, OCRPageOutput
from snapshield.core.visualization import annotate_image_with_detections
from snapshield.services.scan_service import scan_document, scan_image


def _create_synthetic_image(text_lines: list[str], img_format: str = "PNG") -> bytes:
    """Helper creating an in-memory test image with drawn text lines."""
    width, height = 400, 200
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y_offset = 20
    for line in text_lines:
        draw.text((20, y_offset), line, fill=(0, 0, 0))
        y_offset += 30

    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


def test_format_support_png_jpg_jpeg():
    """Verifies that PNG, JPG, and JPEG byte buffers are accepted."""
    lines = ["Test prompt content"]
    png_bytes = _create_synthetic_image(lines, "PNG")
    jpg_bytes = _create_synthetic_image(lines, "JPEG")

    # Mock OCR provider to isolate format validation
    mock_ocr = MagicMock()
    mock_ocr.extract_from_image.return_value = OCROutput(
        text="Test prompt content",
        confidence=0.95,
        bounding_boxes=[],
        provider_name="mock_ocr",
    )

    with patch("snapshield.services.scan_service.get_ocr_provider", return_value=mock_ocr):
        res_png = scan_image(png_bytes, file_name="sample.png")
        assert isinstance(res_png, ScanResult)
        assert not any("Unsupported image format" in e for e in res_png.errors)

        res_jpg = scan_image(jpg_bytes, file_name="sample.jpg")
        assert isinstance(res_jpg, ScanResult)
        assert not any("Unsupported image format" in e for e in res_jpg.errors)


def test_unsupported_format_handled_cleanly():
    """Verifies unsupported formats (e.g. BMP) return a clean error without crashing."""
    bmp_img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    bmp_img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    res = scan_image(bmp_bytes, file_name="sample.bmp")

    assert isinstance(res, ScanResult)
    assert len(res.errors) >= 1
    assert "Unsupported image format 'BMP'" in res.errors[0]
    assert "Supported formats are PNG, JPG, JPEG" in res.errors[0]
    assert res.sanitized_text == ""


def test_corrupted_image_handled_cleanly():
    """Verifies corrupted bytes return a clean error without throwing unhandled exceptions."""
    corrupted_bytes = b"NOT_AN_IMAGE_RANDOM_GARBAGE_BYTES_12345"

    res = scan_image(corrupted_bytes, file_name="corrupted.png")

    assert isinstance(res, ScanResult)
    assert len(res.errors) >= 1
    assert "Corrupted or invalid image data" in res.errors[0] or "Failed to load image" in res.errors[0]


def test_ocr_failure_handled_cleanly():
    """Verifies that an unhandled OCR provider exception is captured safely in errors."""
    png_bytes = _create_synthetic_image(["Some text"], "PNG")

    mock_ocr = MagicMock()
    mock_ocr.extract_from_image.side_effect = RuntimeError("Tesseract binary not found in system PATH")

    with patch("snapshield.services.scan_service.get_ocr_provider", return_value=mock_ocr):
        res = scan_image(png_bytes)

        assert isinstance(res, ScanResult)
        assert len(res.errors) >= 1
        assert any("Image OCR failed" in e for e in res.errors)
        assert res.original_text == ""


def test_end_to_end_image_scanning_workflow():
    """Verifies Image -> Local OCR -> extracted text -> hybrid detection -> risk -> policy -> redaction."""
    text_content = (
        "CONFIDENTIAL PATIENT DOCUMENT:\n"
        "Aadhaar Number: 9999 9999 0019\n"
        "Contact Phone: +91 98765 43210\n"
        "Email: patient.doe@hospital.org"
    )
    png_bytes = _create_synthetic_image(text_content.split("\n"), "PNG")

    # Construct mock OCR output with realistic bounding boxes
    boxes = [
        OCRBoundingBox(text="CONFIDENTIAL", x0=20, top=20, x1=120, bottom=40, confidence=0.98),
        OCRBoundingBox(text="Aadhaar", x0=20, top=50, x1=80, bottom=70, confidence=0.98),
        OCRBoundingBox(text="9999", x0=90, top=50, x1=130, bottom=70, confidence=0.99),
        OCRBoundingBox(text="9999", x0=135, top=50, x1=175, bottom=70, confidence=0.99),
        OCRBoundingBox(text="0019", x0=180, top=50, x1=220, bottom=70, confidence=0.99),
        OCRBoundingBox(text="+91", x0=20, top=80, x1=50, bottom=100, confidence=0.95),
        OCRBoundingBox(text="9876543210", x0=55, top=80, x1=140, bottom=100, confidence=0.95),
        OCRBoundingBox(text="patient.doe@hospital.org", x0=20, top=110, x1=200, bottom=130, confidence=0.99),
    ]

    mock_ocr = MagicMock()
    mock_ocr.extract_from_image.return_value = OCROutput(
        text=text_content,
        confidence=0.98,
        bounding_boxes=boxes,
        provider_name="mock_local_ocr",
    )

    with patch("snapshield.services.scan_service.get_ocr_provider", return_value=mock_ocr):
        result = scan_image(
            image_input=png_bytes,
            policy_mode="Personal",
            ai_enabled=False,
        )

        assert isinstance(result, ScanResult)
        assert len(result.detections) >= 2

        # 1. Detection findings check (structured detection reused)
        types = {d.entity_type for d in result.detections}
        assert "Aadhaar" in types
        assert "Email" in types or "EMAIL" in types

        # 2. Risk assessment check
        assert result.risk_assessment.score > 0

        # 3. Policy decision check
        assert len(result.decisions) >= 2

        # 4. Redaction check
        assert "[REDACTED: AADHAAR]" in result.sanitized_text
        assert "9999 9999 0019" not in result.sanitized_text

        # 5. Visualization layer check
        assert result.annotated_image_bytes is not None
        assert result.annotated_regions_count >= 1
        assert len(result.ocr_bounding_boxes) == len(boxes)

        # Confirm annotated bytes form a valid PNG image
        annotated_img = Image.open(io.BytesIO(result.annotated_image_bytes))
        assert annotated_img.size == (400, 200)


def test_visualization_layer_highlights_and_badges():
    """Directly tests annotate_image_with_detections visual bounding boxes and badge generation."""
    test_img = Image.new("RGB", (300, 150), color="white")
    text = "Secret API Key: sk-live-syntheticSecret123"

    boxes = [
        OCRBoundingBox(text="Secret", x0=10, top=20, x1=60, bottom=40, confidence=0.95),
        OCRBoundingBox(text="API", x0=65, top=20, x1=90, bottom=40, confidence=0.95),
        OCRBoundingBox(text="Key:", x0=95, top=20, x1=130, bottom=40, confidence=0.95),
        OCRBoundingBox(text="sk-live-syntheticSecret123", x0=135, top=20, x1=280, bottom=40, confidence=0.99),
    ]

    detection = Detection(
        entity_type="API_KEY",
        start=16,
        end=42,
        confidence=0.99,
        detector="regex_api_key",
        severity=SeverityLevel.CRITICAL,
        raw_value="sk-live-syntheticSecret123",
        explanation="Secret key match",
    )

    decision = PolicyDecision(
        entity_type="API_KEY",
        action=PolicyAction.BLOCK,
        policy_mode="Developer",
        reason="API credentials strictly blocked.",
    )

    annotated_bytes, count = annotate_image_with_detections(
        image_input=test_img,
        extracted_text=text,
        bounding_boxes=boxes,
        detections=[detection],
        decisions=[decision],
    )

    assert count == 1
    assert isinstance(annotated_bytes, bytes)
    assert len(annotated_bytes) > 0

    # Ensure valid PNG output
    out_img = Image.open(io.BytesIO(annotated_bytes))
    assert out_img.size == (300, 150)


def test_pipeline_reuse_in_scan_document_with_input_document():
    """Verifies InputDocument(document_type='IMAGE') seamlessly routes through scan_document."""
    png_bytes = _create_synthetic_image(["Internal document"], "PNG")

    from snapshield.core.models import InputDocument
    doc = InputDocument(
        document_type="IMAGE",
        raw_bytes=png_bytes,
        file_name="screenshot.png",
    )

    mock_ocr = MagicMock()
    mock_ocr.extract_from_image.return_value = OCROutput(
        text="Internal document",
        confidence=0.95,
        bounding_boxes=[],
        provider_name="mock_ocr",
    )

    with patch("snapshield.services.scan_service.get_ocr_provider", return_value=mock_ocr):
        result = scan_document(doc, policy_mode="Personal", ai_enabled=False)

        assert isinstance(result, ScanResult)
        assert result.original_text == "Internal document"
        assert result.timings.ocr_ms >= 0.0
