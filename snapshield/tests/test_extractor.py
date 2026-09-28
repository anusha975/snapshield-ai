"""Comprehensive tests for local PDF text and layout extraction.

Tests:
1. Normal PDFs (multi-page text extraction):
   - Page-by-page extraction
   - Preservation of page boundaries (--- Page X ---)
   - Preservation of word positions (x0, top, x1, bottom)
   - Character and word counting
   - OCR-not-required status for rich text PDFs
2. Empty PDFs (blank pages):
   - Detection of empty PDF status
   - OCR-not-required for empty pages
3. Scanned PDFs (image-only):
   - Detection of scanned PDF status
   - Accurate OCR-required flagging
4. Malformed & Corrupt PDFs:
   - Non-PDF signatures rejected
   - Truncated / corrupt byte streams rejected
   - Zero-byte input rejected
   - None input rejected
5. Size limits:
   - Custom max size enforcement
"""

from __future__ import annotations

import io
import pytest
from PIL import Image
import pymupdf

from snapshield.core.exceptions import ExtractionError
from snapshield.core.extractor import extract_pdf
from snapshield.core.models import InputDocument


# ==========================================
# Helpers for PDF Generation
# ==========================================

def make_text_pdf(pages_text: list[str]) -> bytes:
    """Generates an in-memory PDF with specified text on each page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=400, height=300)
        page.insert_text((50, 72), text)
    return doc.tobytes()


def make_blank_pdf(page_count: int = 1) -> bytes:
    """Generates a blank PDF with empty pages."""
    doc = pymupdf.open()
    for _ in range(page_count):
        doc.new_page(width=400, height=300)
    return doc.tobytes()


def make_scanned_pdf(page_count: int = 1) -> bytes:
    """Generates a scanned PDF containing image objects with zero embedded text."""
    doc = pymupdf.open()
    # Create sample image
    img = Image.new("RGB", (120, 80), color=(30, 144, 255))
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    for _ in range(page_count):
        page = doc.new_page(width=400, height=300)
        page.insert_image(page.rect, stream=img_bytes)
    return doc.tobytes()


# ==========================================
# 1. Normal Multi-Page PDF Tests
# ==========================================

def test_extract_normal_multi_page_pdf():
    """Verify normal PDF extracts page-by-page, preserving boundaries and word positions."""
    pages_content = [
        "Patient John Doe SSN 123-45-6789.",
        "Clinical diagnosis: hypertension and asthma.",
    ]
    pdf_bytes = make_text_pdf(pages_content)

    doc = extract_pdf(pdf_bytes, file_name="medical_records.pdf")

    assert isinstance(doc, InputDocument)
    assert doc.document_type == "PDF"
    assert doc.file_name == "medical_records.pdf"
    assert doc.file_size_bytes == len(pdf_bytes)

    # Verify page boundary preservation
    assert "--- Page 1 ---" in doc.raw_content
    assert "--- Page 2 ---" in doc.raw_content
    assert "Patient John Doe" in doc.raw_content
    assert "Clinical diagnosis" in doc.raw_content

    # Metadata assertions
    meta = doc.metadata
    assert meta["page_count"] == 2
    assert meta["ocr_required"] is False
    assert meta["is_scanned_pdf"] is False
    assert meta["is_empty_pdf"] is False
    assert meta["total_chars"] > 20
    assert meta["total_words"] >= 8

    # Word position assertions
    page1 = meta["pages"][0]
    assert page1["page_number"] == 1
    assert page1["has_text"] is True
    assert len(page1["positions"]) > 0

    first_word = page1["positions"][0]
    assert "text" in first_word
    assert "x0" in first_word
    assert "top" in first_word
    assert "x1" in first_word
    assert "bottom" in first_word
    assert first_word["x1"] >= first_word["x0"]


def test_extract_pdf_file_like_stream():
    """Verify extract_pdf handles io.BytesIO stream cleanly."""
    pdf_bytes = make_text_pdf(["Single page stream test."])
    stream = io.BytesIO(pdf_bytes)

    doc = extract_pdf(stream, file_name="stream.pdf")
    assert doc.metadata["page_count"] == 1
    assert "Single page stream test." in doc.raw_content


# ==========================================
# 2. Empty PDF Tests
# ==========================================

def test_extract_empty_pdf():
    """Verify blank PDF is detected as empty and does not require OCR."""
    pdf_bytes = make_blank_pdf(page_count=2)

    doc = extract_pdf(pdf_bytes, file_name="blank.pdf")
    assert doc.metadata["page_count"] == 2
    assert doc.metadata["is_empty_pdf"] is True
    assert doc.metadata["is_scanned_pdf"] is False
    assert doc.metadata["ocr_required"] is False
    assert doc.metadata["total_chars"] == 0
    assert doc.raw_content == ""


# ==========================================
# 3. Scanned (Image-Only) PDF Tests
# ==========================================

def test_extract_scanned_image_only_pdf():
    """Verify scanned PDF flags is_scanned_pdf=True and ocr_required=True."""
    pdf_bytes = make_scanned_pdf(page_count=1)

    doc = extract_pdf(pdf_bytes, file_name="scanned_receipt.pdf")
    assert doc.metadata["page_count"] == 1
    assert doc.metadata["is_scanned_pdf"] is True
    assert doc.metadata["ocr_required"] is True
    assert doc.metadata["is_empty_pdf"] is False
    assert doc.metadata["total_images"] >= 1
    assert doc.metadata["total_chars"] == 0


# ==========================================
# 4. Malformed and Invalid PDF Tests
# ==========================================

def test_extract_non_pdf_signature_raises():
    """Verify that input without %PDF- header raises ExtractionError."""
    fake_bytes = b"Hello, this is not a PDF file at all!"
    with pytest.raises(ExtractionError) as exc:
        extract_pdf(fake_bytes)
    assert "does not have a valid PDF header signature" in str(exc.value)


def test_extract_corrupted_pdf_raises():
    """Verify truncated or corrupted PDF structure raises ExtractionError."""
    corrupted_bytes = b"%PDF-1.4\nSomeCorruptedDataHereThatCannotBeParsed"
    with pytest.raises(ExtractionError) as exc:
        extract_pdf(corrupted_bytes)
    assert "Failed to extract text from malformed or encrypted PDF" in str(exc.value)


def test_extract_empty_byte_stream_raises():
    """Verify zero-byte PDF input raises ExtractionError."""
    with pytest.raises(ExtractionError) as exc:
        extract_pdf(b"")
    assert "PDF file is empty (0 bytes)" in str(exc.value)


def test_extract_none_input_raises():
    """Verify None input raises ExtractionError."""
    with pytest.raises(ExtractionError) as exc:
        extract_pdf(None)  # type: ignore
    assert "PDF input cannot be None" in str(exc.value)


# ==========================================
# 5. Size Limit Enforcement Tests
# ==========================================

def test_extract_pdf_size_limit_enforced():
    """Verify size limits reject oversized PDF inputs."""
    pdf_bytes = make_text_pdf(["Small document."])
    # Enforce a tiny max limit of 100 bytes
    with pytest.raises(ExtractionError) as exc:
        extract_pdf(pdf_bytes, max_size_bytes=100)
    assert "exceeds maximum allowed limit" in str(exc.value)
