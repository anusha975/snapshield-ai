"""Unified Input Layer and Document Normalization for SnapShield AI.

Supports:
1. Pasted text
2. TXT files
3. PDF documents
4. PNG images
5. JPG / JPEG images

Provides:
- Strict file extension validation
- MIME type and magic-bytes verification (prevents extension spoofing)
- Bounded file size verification (enforcing MAX_FILE_SIZE_MB)
- Malformed input rejection
- Safe in-memory processing (no unsafe file writes or directory traversals)
- Zero raw content logging (enforces Rule 2)
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import pdfplumber

from snapshield.config.settings import get_settings
from snapshield.core.exceptions import InputValidationError
from snapshield.core.logger import logger
from snapshield.core.models import InputDocument


# ==========================================
# Supported Specifications & Magic Headers
# ==========================================

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}

MIME_TYPE_MAPPING = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Magic byte signatures
MAGIC_SIGNATURES = {
    ".pdf": [b"%PDF-"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
}


class FileValidationResult(BaseModel):
    """Result of validating an uploaded file."""
    is_valid: bool
    file_name: str
    file_size_bytes: int = Field(ge=0)
    file_extension: str
    detected_mime_type: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==========================================
# Extraction Helpers
# Configure PIL decompression bomb protection limit (25 Megapixels)
Image.MAX_IMAGE_PIXELS = 25_000_000

MAX_IMAGE_DIMENSION = 8192
MAX_PDF_PAGES = 100


def sanitize_filename(raw_filename: str) -> str:
    """Sanitizes filename against path traversal, null bytes, and malicious control characters."""
    if not raw_filename or not isinstance(raw_filename, str):
        return "uploaded_document"

    # Remove null bytes and control characters
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw_filename).strip()

    # Eliminate directory traversal patterns
    cleaned = re.sub(r"\.\.+[/\\]", "", cleaned)

    # Use pure filename basename
    cleaned = Path(cleaned).name

    # Strip dangerous leading/trailing dots or spaces
    cleaned = cleaned.strip(". ")

    if not cleaned:
        return "uploaded_document"

    return cleaned


def _extract_file_info(file_input: Any) -> Tuple[str, bytes]:
    """Safely extracts the file name and byte payload from various file inputs.

    Supports Streamlit UploadedFile, file-like objects, Path, str, or raw bytes.
    Does NOT write to filesystem.
    """
    if file_input is None:
        raise InputValidationError("No file input was provided (received None).")

    # Raw bytes with no filename
    if isinstance(file_input, bytes):
        return "upload.bin", file_input

    # File path (str or Path)
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not path.is_file():
            raise InputValidationError(f"File path does not exist or is not a regular file: {path.name}")
        return sanitize_filename(path.name), path.read_bytes()

    # Streamlit UploadedFile or file-like object
    raw_name = getattr(file_input, "name", "uploaded_document")
    file_name = sanitize_filename(raw_name)

    if hasattr(file_input, "getvalue"):
        data = file_input.getvalue()
    elif hasattr(file_input, "read"):
        if hasattr(file_input, "seek"):
            file_input.seek(0)
        data = file_input.read()
        if hasattr(file_input, "seek"):
            file_input.seek(0)
    else:
        raise InputValidationError(f"Unsupported file object type: {type(file_input).__name__}")

    return file_name, data


# ==========================================
# Public API
# ==========================================

def create_text_document(text: str, file_name: Optional[str] = None) -> InputDocument:
    """Creates a normalized InputDocument from a plain text string or prompt.

    Enforces:
    - Text type validation
    - Non-empty validation
    - Safe logging (never logs the raw text content)
    """
    if not isinstance(text, str):
        raise InputValidationError(
            f"Expected text input to be a string, got {type(text).__name__}."
        )

    stripped = text.strip()
    if not stripped:
        raise InputValidationError("Text input cannot be empty or purely whitespace.")

    size_bytes = len(text.encode("utf-8"))
    settings = get_settings()
    if size_bytes > settings.max_file_size_bytes:
        raise InputValidationError(
            f"Text length exceeds maximum allowed payload size of {settings.max_file_size_mb} MB.",
            details={"size_bytes": size_bytes, "max_bytes": settings.max_file_size_bytes},
        )

    doc = InputDocument(
        document_type="RAW_TEXT",
        raw_content=text,
        raw_bytes=text.encode("utf-8"),
        file_name=file_name or "pasted_text.txt",
        file_size_bytes=size_bytes,
        metadata={"source": "pasted_text", "char_count": len(text)},
    )

    logger.info("Created text document [id=%s, size=%d bytes]", doc.document_id, doc.file_size_bytes)
    return doc


def validate_uploaded_file(file: Any) -> FileValidationResult:
    """Validates an uploaded file across extension, size, MIME type, and structure.

    Returns FileValidationResult with is_valid=True or descriptive error_message.
    Does NOT raise exceptions for validation failures, enabling clean error reporting.
    """
    try:
        file_name, data = _extract_file_info(file)
    except InputValidationError as exc:
        return FileValidationResult(
            is_valid=False,
            file_name=getattr(file, "name", "unknown"),
            file_size_bytes=0,
            file_extension="",
            detected_mime_type="application/octet-stream",
            error_message=str(exc),
        )

    size_bytes = len(data)
    settings = get_settings()

    # 1. Size Validation
    if size_bytes == 0:
        return FileValidationResult(
            is_valid=False,
            file_name=file_name,
            file_size_bytes=0,
            file_extension=Path(file_name).suffix.lower(),
            detected_mime_type="application/octet-stream",
            error_message="Uploaded file is empty (0 bytes).",
        )

    if size_bytes > settings.max_file_size_bytes:
        size_mb = size_bytes / (1024 * 1024)
        return FileValidationResult(
            is_valid=False,
            file_name=file_name,
            file_size_bytes=size_bytes,
            file_extension=Path(file_name).suffix.lower(),
            detected_mime_type="application/octet-stream",
            error_message=(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of "
                f"{settings.max_file_size_mb} MB."
            ),
        )

    # 2. Extension Validation
    ext = Path(file_name).suffix.lower()
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        return FileValidationResult(
            is_valid=False,
            file_name=file_name,
            file_size_bytes=size_bytes,
            file_extension=ext or "[none]",
            detected_mime_type="application/octet-stream",
            error_message=(
                f"Unsupported file extension '{ext}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    expected_mime = MIME_TYPE_MAPPING[ext]

    # 3. Magic Bytes / Header Validation
    if ext in MAGIC_SIGNATURES:
        signatures = MAGIC_SIGNATURES[ext]
        if not any(data.startswith(sig) for sig in signatures):
            return FileValidationResult(
                is_valid=False,
                file_name=file_name,
                file_size_bytes=size_bytes,
                file_extension=ext,
                detected_mime_type="application/octet-stream",
                error_message=(
                    f"File extension is '{ext}', but file contents do not match "
                    f"valid {expected_mime} header signatures (possible extension spoofing)."
                ),
            )

    # 4. Malformed Content Verification
    metadata: Dict[str, Any] = {"mime_type": expected_mime}

    if ext == ".txt":
        try:
            # Check for non-text binary data or decoding failure
            if b"\x00" in data[:1024]:
                return FileValidationResult(
                    is_valid=False,
                    file_name=file_name,
                    file_size_bytes=size_bytes,
                    file_extension=ext,
                    detected_mime_type="application/octet-stream",
                    error_message="TXT file contains binary null bytes, indicating corrupted or binary payload.",
                )
            decoded_text = data.decode("utf-8")
            metadata["char_count"] = len(decoded_text)
        except UnicodeDecodeError:
            try:
                decoded_text = data.decode("latin-1")
                metadata["char_count"] = len(decoded_text)
                metadata["encoding"] = "latin-1"
            except Exception as dec_exc:
                return FileValidationResult(
                    is_valid=False,
                    file_name=file_name,
                    file_size_bytes=size_bytes,
                    file_extension=ext,
                    detected_mime_type="application/octet-stream",
                    error_message=f"Failed to decode text file with supported encodings: {dec_exc}",
                )

    elif ext == ".pdf":
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                page_count = len(pdf.pages)
                metadata["page_count"] = page_count
                if page_count == 0:
                    return FileValidationResult(
                        is_valid=False,
                        file_name=file_name,
                        file_size_bytes=size_bytes,
                        file_extension=ext,
                        detected_mime_type=expected_mime,
                        error_message="PDF document contains zero pages.",
                    )
                if page_count > MAX_PDF_PAGES:
                    return FileValidationResult(
                        is_valid=False,
                        file_name=file_name,
                        file_size_bytes=size_bytes,
                        file_extension=ext,
                        detected_mime_type=expected_mime,
                        error_message=f"PDF exceeds safety threshold of {MAX_PDF_PAGES} pages (contains {page_count} pages).",
                    )
        except Exception as pdf_exc:
            return FileValidationResult(
                is_valid=False,
                file_name=file_name,
                file_size_bytes=size_bytes,
                file_extension=ext,
                detected_mime_type=expected_mime,
                error_message=f"Malformed or corrupted PDF document: {pdf_exc}",
            )

    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()  # Verifies file integrity
                w, h = img.width, img.height
                metadata["image_format"] = img.format
                metadata["image_dimensions"] = (w, h)
                if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                    return FileValidationResult(
                        is_valid=False,
                        file_name=file_name,
                        file_size_bytes=size_bytes,
                        file_extension=ext,
                        detected_mime_type=expected_mime,
                        error_message=(
                            f"Image dimensions ({w}x{h}) exceed maximum allowed dimension limit "
                            f"of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels (decompression bomb protection)."
                        ),
                    )
        except (UnidentifiedImageError, OSError, Exception) as img_exc:
            return FileValidationResult(
                is_valid=False,
                file_name=file_name,
                file_size_bytes=size_bytes,
                file_extension=ext,
                detected_mime_type=expected_mime,
                error_message=f"Corrupted or unreadable image file: {img_exc}",
            )

    logger.info("File validation passed for '%s' (MIME: %s, size: %d bytes)",
                file_name, expected_mime, size_bytes)

    return FileValidationResult(
        is_valid=True,
        file_name=file_name,
        file_size_bytes=size_bytes,
        file_extension=ext,
        detected_mime_type=expected_mime,
        metadata=metadata,
    )


def create_document_from_upload(file: Any) -> InputDocument:
    """Validates an uploaded file and produces a normalized InputDocument.

    Raises:
        InputValidationError: If the file is invalid, unsupported, or malformed.
    """
    validation = validate_uploaded_file(file)
    if not validation.is_valid:
        raise InputValidationError(
            validation.error_message or "Uploaded file validation failed.",
            details={
                "file_name": validation.file_name,
                "file_extension": validation.file_extension,
                "file_size_bytes": validation.file_size_bytes,
            },
        )

    file_name, data = _extract_file_info(file)
    ext = validation.file_extension

    raw_content: Optional[str] = None
    doc_type: str = "UNKNOWN"

    if ext == ".txt":
        doc_type = "TXT"
        try:
            raw_content = data.decode("utf-8")
        except UnicodeDecodeError:
            raw_content = data.decode("latin-1")

    elif ext == ".pdf":
        from snapshield.core.extractor import extract_pdf
        return extract_pdf(data, file_name=file_name)

    elif ext in [".png", ".jpg", ".jpeg"]:
        doc_type = "IMAGE"
        # Image raw_content is None until OCR is executed
        raw_content = None

    doc = InputDocument(
        document_type=doc_type,
        raw_content=raw_content,
        raw_bytes=data,
        file_name=file_name,
        file_size_bytes=len(data),
        metadata=validation.metadata,
    )

    logger.info("Successfully created normalized InputDocument [id=%s, type=%s, name=%s, size=%d bytes]",
                doc.document_id, doc.document_type, doc.file_name, doc.file_size_bytes)
    return doc
