"""Comprehensive tests for replaceable local OCR architecture and synthetic image processing.

Validates:
1. OCR output schemas:
   - Text, confidence, bounding boxes (with x0, top, x1, bottom), and page index
2. Decoupling:
   - OCR output feeds into HybridDetectionEngine just like normal text
3. LocalOCRProvider:
   - Synthetic image processing
   - Graceful failure handling when binary is absent or image is corrupted
4. Replaceable Architecture:
   - OCRRegistry registration and swapping with accelerated/custom providers
   - Multi-page image consolidation
"""

from __future__ import annotations

import io
import pytest
from PIL import Image, ImageDraw

from snapshield.core.exceptions import OCRError
from snapshield.core.ocr import (
    AcceleratedOCRProvider,
    LocalOCRProvider,
    OCRBoundingBox,
    OCROutput,
    OCRPageOutput,
    OCRProvider,
    OCRRegistry,
    get_ocr_provider,
)
from snapshield.services.detector import HybridDetectionEngine


# ==========================================
# Helpers for Synthetic Images
# ==========================================

def create_synthetic_text_image(text_lines: list[str]) -> Image.Image:
    """Generates an in-memory synthetic image with rendered text lines."""
    width = 400
    height = max(100, len(text_lines) * 35 + 40)
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    y = 20
    for line in text_lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 30

    return image


def create_synthetic_ocr_fallback(
    text: str = "Synthetic Patient: Jane Doe SSN: 219-45-6789",
    confidence: float = 0.94,
):
    """Produces a mock OCR extractor function simulating local OCR engine output."""
    def fallback(img: Image.Image, page_idx: int) -> OCROutput:
        words = text.split()
        boxes = []
        cur_x = 20.0
        for w in words:
            w_len = len(w) * 8.0
            boxes.append(
                OCRBoundingBox(
                    text=w,
                    x0=cur_x,
                    top=25.0,
                    x1=cur_x + w_len,
                    bottom=45.0,
                    confidence=confidence,
                    page_index=page_idx,
                )
            )
            cur_x += w_len + 10.0

        page_out = OCRPageOutput(
            page_index=page_idx,
            text=text,
            confidence=confidence,
            bounding_boxes=boxes,
            char_count=len(text),
            word_count=len(words),
        )

        return OCROutput(
            text=text,
            confidence=confidence,
            bounding_boxes=boxes,
            pages=[page_out],
            provider_name="mock_synthetic_ocr",
            is_accelerated=False,
        )

    return fallback


# ==========================================
# 1. OCR Output Schema Tests
# ==========================================

def test_ocr_output_schema_and_conversion():
    """Verify OCROutput contains text, confidence, bounding boxes, and converts to model."""
    box = OCRBoundingBox(
        text="SampleToken",
        x0=10.0,
        top=15.0,
        x1=80.0,
        bottom=35.0,
        confidence=0.91,
        page_index=0,
    )

    page = OCRPageOutput(
        page_index=0,
        text="SampleToken",
        confidence=0.91,
        bounding_boxes=[box],
        char_count=11,
        word_count=1,
    )

    output = OCROutput(
        text="SampleToken",
        confidence=0.91,
        bounding_boxes=[box],
        pages=[page],
        provider_name="test_ocr_provider",
        processing_time_ms=15.2,
    )

    assert output.text == "SampleToken"
    assert output.confidence == 0.91
    assert len(output.bounding_boxes) == 1
    assert output.bounding_boxes[0].page_index == 0
    assert output.to_detection_input() == "SampleToken"

    # Test conversion to standard core OCRResult
    model_res = output.to_model()
    assert model_res.text == "SampleToken"
    assert model_res.confidence == 0.91
    assert model_res.engine == "test_ocr_provider"
    assert model_res.char_count == 11


# ==========================================
# 2. Local OCR Provider & Synthetic Images
# ==========================================

def test_local_ocr_with_synthetic_image():
    """Verify LocalOCRProvider processes synthetic images and yields structured text & boxes."""
    synthetic_text = "CONFIDENTIAL: Developer API Key sk-syntheticSecretKey12345678"
    fallback_fn = create_synthetic_ocr_fallback(text=synthetic_text, confidence=0.96)

    provider = LocalOCRProvider(custom_fallback=fallback_fn)
    assert provider.is_available() is True
    assert provider.is_accelerated is False

    img = create_synthetic_text_image([synthetic_text])
    res = provider.extract_from_image(img, page_index=1)

    assert res.text == synthetic_text
    assert res.confidence == 0.96
    assert len(res.bounding_boxes) > 0
    assert res.bounding_boxes[0].page_index == 1
    assert len(res.pages) == 1
    assert res.pages[0].page_index == 1


def test_ocr_output_consumed_by_detection_engine():
    """Verify OCR output is directly consumable by HybridDetectionEngine like normal text."""
    ocr_text = "Billing alert: credit card 4532-0150-1234-5671 charged $200. Contact support@synthetic-corp.io"
    fallback_fn = create_synthetic_ocr_fallback(text=ocr_text, confidence=0.98)

    provider = LocalOCRProvider(custom_fallback=fallback_fn)
    ocr_result = provider.extract_from_image(create_synthetic_text_image([ocr_text]))

    # Detection engine consumes OCR output text just like normal text
    detector = HybridDetectionEngine()
    findings = detector.scan_text(ocr_result.to_detection_input())

    assert len(findings) >= 2
    categories = [f.entity_category.value for f in findings]
    assert "PAYMENT_CARD" in categories
    assert "CONTACT_INFO" in categories


def test_multi_page_ocr_consolidation():
    """Verify extract_from_pages consolidates multiple images with page boundaries."""
    img1 = create_synthetic_text_image(["Page One Statement"])
    img2 = create_synthetic_text_image(["Page Two Statement"])

    call_count = 0

    def multi_fallback(img, page_idx):
        nonlocal call_count
        call_count += 1
        return create_synthetic_ocr_fallback(text=f"Content from page {page_idx}", confidence=0.90)(img, page_idx)

    provider = LocalOCRProvider(custom_fallback=multi_fallback)
    consolidated = provider.extract_from_pages([img1, img2])

    assert call_count == 2
    assert len(consolidated.pages) == 2
    assert "--- Page 1 ---" in consolidated.text
    assert "--- Page 2 ---" in consolidated.text
    assert consolidated.confidence == pytest.approx(0.90, 0.01)


# ==========================================
# 3. Graceful Failure & Error Handling
# ==========================================

def test_ocr_missing_binary_raises_gracefully():
    """Verify that when local binary is absent and no fallback exists, OCRError is raised."""
    # Point to an invalid tesseract binary path
    provider = LocalOCRProvider(tesseract_cmd="C:\\nonexistent_tesseract_binary.exe")
    assert provider.is_available() is False

    img = create_synthetic_text_image(["Some text"])
    with pytest.raises(OCRError) as exc:
        provider.extract_from_image(img)
    assert "Local Tesseract OCR binary not detected" in str(exc.value)


def test_ocr_corrupted_image_raises_gracefully():
    """Verify corrupted image input raises OCRError instead of unhandled crash."""
    provider = LocalOCRProvider(custom_fallback=create_synthetic_ocr_fallback())
    corrupted_data = b"CorruptedNonImageDataStream"

    with pytest.raises(OCRError) as exc:
        provider.extract_from_image(corrupted_data)
    assert "Failed to load or parse image for OCR" in str(exc.value)


# ==========================================
# 4. Replaceable Registry & Acceleration Stubs
# ==========================================

def test_ocr_registry_pluggable_architecture():
    """Verify OCR providers can be swapped cleanly via OCRRegistry."""
    original_provider = get_ocr_provider()

    class CustomTestProvider(OCRProvider):
        @property
        def name(self) -> str:
            return "custom_enterprise_ocr"

        def is_available(self) -> bool:
            return True

        def extract_from_image(self, image_input, page_index: int = 0) -> OCROutput:
            return OCROutput(
                text="Custom extracted text",
                confidence=0.99,
                bounding_boxes=[],
                pages=[],
                provider_name=self.name,
            )

    custom_provider = CustomTestProvider()
    OCRRegistry.set_provider(custom_provider)

    active = get_ocr_provider()
    assert active.name == "custom_enterprise_ocr"
    res = active.extract_from_image(Image.new("RGB", (10, 10)))
    assert res.text == "Custom extracted text"

    # Restore original provider
    OCRRegistry.set_provider(original_provider)


def test_accelerated_ocr_provider_stub():
    """Verify AcceleratedOCRProvider reports truthful acceleration status."""
    accel_provider = AcceleratedOCRProvider(model_path="models/ocr_dbnet.onnx", active_backend="QNNExecutionProvider")
    assert accel_provider.is_accelerated is True
    assert "QNNExecutionProvider" in accel_provider.name
    assert accel_provider.is_available() is True

    cpu_provider = AcceleratedOCRProvider(model_path="models/ocr_dbnet.onnx", active_backend="CPU")
    assert cpu_provider.is_accelerated is False
