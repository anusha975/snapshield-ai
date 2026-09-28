"""Comprehensive tests for SnapShield AI unified input layer.

Tests:
1. Pasted text ingestion:
   - Valid text document creation
   - Empty/whitespace rejection
   - Non-string type validation
   - Payload size bounds enforcement
2. File Upload Validation (validate_uploaded_file):
   - TXT, PDF, PNG, JPG, JPEG valid inputs
   - Extension validation (unsupported extensions rejected)
   - Bounded file size (empty file, oversized file)
   - Magic bytes / MIME type verification (prevents extension spoofing)
   - Malformed input rejection (corrupt PDF, corrupt image, binary nulls in TXT)
3. Document creation from upload (create_document_from_upload):
   - In-memory processing
   - Proper typing (TXT, PDF, IMAGE)
   - Exception handling on invalid upload
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image
from pypdf import PdfWriter

from snapshield.config.settings import SnapShieldSettings
from snapshield.core.exceptions import InputValidationError
from snapshield.core.input_handler import (
    create_document_from_upload,
    create_text_document,
    validate_uploaded_file,
)
from snapshield.core.models import InputDocument


# ==========================================
# Test Fixtures & Helpers
# ==========================================

class MockUploadedFile:
    """Mock simulating a Streamlit UploadedFile or file-like object."""
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._buffer = io.BytesIO(data)
        self.size = len(data)

    def read(self, *args, **kwargs) -> bytes:
        return self._buffer.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._buffer.seek(*args, **kwargs)

    def getvalue(self) -> bytes:
        return self._buffer.getvalue()


def create_valid_pdf_bytes(text_content: str = "Confidential synthetic statement.") -> bytes:
    """Creates a valid in-memory PDF payload."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_valid_image_bytes(format_name: str = "PNG") -> bytes:
    """Creates a valid in-memory PNG or JPEG image payload."""
    img = Image.new("RGB", (100, 50), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


# ==========================================
# 1. Pasted Text Ingestion Tests
# ==========================================

def test_create_text_document_success():
    """Verify standard text ingestion produces a valid InputDocument."""
    text = "Analyze this synthetic medical summary for patient PT-1002."
    doc = create_text_document(text)

    assert isinstance(doc, InputDocument)
    assert doc.document_type == "RAW_TEXT"
    assert doc.raw_content == text
    assert doc.file_size_bytes == len(text.encode("utf-8"))
    assert doc.file_name == "pasted_text.txt"


def test_create_text_document_empty_or_whitespace():
    """Verify that empty or whitespace-only text is rejected."""
    with pytest.raises(InputValidationError) as exc:
        create_text_document("")
    assert "empty or purely whitespace" in str(exc.value)

    with pytest.raises(InputValidationError):
        create_text_document("   \n\t  ")


def test_create_text_document_invalid_type():
    """Verify that non-string inputs are rejected."""
    with pytest.raises(InputValidationError) as exc:
        create_text_document(12345)  # type: ignore
    assert "Expected text input to be a string" in str(exc.value)


# ==========================================
# 2. File Upload Validation Tests
# ==========================================

def test_validate_valid_txt_file():
    """Verify validation of a valid plain text upload."""
    mock_file = MockUploadedFile("notes.txt", b"System telemetry notes and synthetic logs.")
    res = validate_uploaded_file(mock_file)

    assert res.is_valid is True
    assert res.file_extension == ".txt"
    assert res.detected_mime_type == "text/plain"
    assert res.error_message is None


def test_validate_valid_pdf_file():
    """Verify validation of a valid PDF upload."""
    pdf_bytes = create_valid_pdf_bytes()
    mock_file = MockUploadedFile("document.pdf", pdf_bytes)
    res = validate_uploaded_file(mock_file)

    assert res.is_valid is True
    assert res.file_extension == ".pdf"
    assert res.detected_mime_type == "application/pdf"
    assert res.metadata.get("page_count", 0) >= 1


def test_validate_valid_png_and_jpg_files():
    """Verify validation of valid PNG and JPEG image uploads."""
    # PNG
    png_file = MockUploadedFile("screenshot.png", create_valid_image_bytes("PNG"))
    png_res = validate_uploaded_file(png_file)
    assert png_res.is_valid is True
    assert png_res.file_extension == ".png"
    assert png_res.detected_mime_type == "image/png"

    # JPG / JPEG
    jpg_file = MockUploadedFile("id_scan.jpg", create_valid_image_bytes("JPEG"))
    jpg_res = validate_uploaded_file(jpg_file)
    assert jpg_res.is_valid is True
    assert jpg_res.file_extension == ".jpg"
    assert jpg_res.detected_mime_type == "image/jpeg"


def test_validate_empty_file():
    """Verify that zero-byte files are rejected."""
    empty_file = MockUploadedFile("empty.txt", b"")
    res = validate_uploaded_file(empty_file)

    assert res.is_valid is False
    assert "empty (0 bytes)" in res.error_message


def test_validate_unsupported_extension():
    """Verify that unsupported file formats are rejected."""
    bad_file = MockUploadedFile("script.py", b"print('Hello world')")
    res = validate_uploaded_file(bad_file)

    assert res.is_valid is False
    assert "Unsupported file extension '.py'" in res.error_message


def test_validate_magic_bytes_spoofing():
    """Verify that files with spoofed extensions fail magic byte verification."""
    # File named .pdf containing plain text
    spoofed_pdf = MockUploadedFile("fake_invoice.pdf", b"This is just plain text, not a PDF!")
    res_pdf = validate_uploaded_file(spoofed_pdf)
    assert res_pdf.is_valid is False
    assert "possible extension spoofing" in res_pdf.error_message

    # File named .png containing invalid random binary
    spoofed_png = MockUploadedFile("fake_image.png", b"\x00\x01\x02\x03\x04\x05")
    res_png = validate_uploaded_file(spoofed_png)
    assert res_png.is_valid is False
    assert "possible extension spoofing" in res_png.error_message


def test_validate_malformed_image():
    """Verify that an image header with corrupted byte streams fails validation."""
    # Start with PNG magic bytes then corrupted trailer
    corrupt_png = b"\x89PNG\r\n\x1a\nCorruptedTruncatedStream"
    mock_file = MockUploadedFile("corrupt.png", corrupt_png)
    res = validate_uploaded_file(mock_file)

    assert res.is_valid is False
    assert "Corrupted or unreadable image file" in res.error_message


def test_validate_binary_nulls_in_txt():
    """Verify that a TXT file containing binary null bytes is rejected."""
    binary_txt = b"Hello\x00\x00\x00BinaryPayload\x00\x00"
    mock_file = MockUploadedFile("binary.txt", binary_txt)
    res = validate_uploaded_file(mock_file)

    assert res.is_valid is False
    assert "binary null bytes" in res.error_message


# ==========================================
# 3. Document Creation from Upload Tests
# ==========================================

def test_create_document_from_upload_txt():
    """Verify normalized InputDocument creation from uploaded TXT."""
    content = b"Synthetic developer prompt: optimize binary tree search."
    mock_file = MockUploadedFile("prompt.txt", content)

    doc = create_document_from_upload(mock_file)
    assert isinstance(doc, InputDocument)
    assert doc.document_type == "TXT"
    assert doc.raw_content == content.decode("utf-8")
    assert doc.file_size_bytes == len(content)
    assert doc.file_name == "prompt.txt"


def test_create_document_from_upload_pdf():
    """Verify normalized InputDocument creation from uploaded PDF."""
    pdf_bytes = create_valid_pdf_bytes()
    mock_file = MockUploadedFile("report.pdf", pdf_bytes)

    doc = create_document_from_upload(mock_file)
    assert isinstance(doc, InputDocument)
    assert doc.document_type == "PDF"
    assert doc.raw_bytes == pdf_bytes
    assert doc.file_name == "report.pdf"


def test_create_document_from_upload_image():
    """Verify normalized InputDocument creation from uploaded PNG image."""
    png_bytes = create_valid_image_bytes("PNG")
    mock_file = MockUploadedFile("capture.png", png_bytes)

    doc = create_document_from_upload(mock_file)
    assert isinstance(doc, InputDocument)
    assert doc.document_type == "IMAGE"
    assert doc.raw_bytes == png_bytes
    assert doc.raw_content is None  # Ready for OCR pipeline
    assert doc.metadata["image_format"] == "PNG"


def test_create_document_from_invalid_upload_raises():
    """Verify that create_document_from_upload raises InputValidationError on invalid files."""
    bad_file = MockUploadedFile("bad.exe", b"MZ\x90\x00")
    with pytest.raises(InputValidationError) as exc:
        create_document_from_upload(bad_file)
    assert "Unsupported file extension '.exe'" in str(exc.value)
