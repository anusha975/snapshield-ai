"""Local PDF Text and Layout Extraction Service for SnapShield AI.

Uses pdfplumber for strictly on-device extraction:
- Page-by-page text extraction preserving page boundaries.
- Positional layout extraction (word bounding boxes).
- Detection of empty vs. scanned (image-only) PDFs.
- Determination of OCR-required status without cloud dependencies.
- Strict size enforcement and malformed PDF error handling.
- Zero raw content logging (enforces Rule 2).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import pdfplumber

from snapshield.config.settings import get_settings
from snapshield.core.exceptions import ExtractionError
from snapshield.core.logger import logger
from snapshield.core.models import InputDocument


class WordPosition(BaseModel):
    """Positional bounding box for an extracted word token."""
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


class PDFPageDetail(BaseModel):
    """Structured extraction metadata for an individual PDF page."""
    page_number: int
    char_count: int
    word_count: int
    has_text: bool
    image_count: int
    has_images: bool
    positions: List[WordPosition] = Field(default_factory=list)


def extract_pdf(
    file_bytes: Union[bytes, BinaryIO, io.BytesIO],
    file_name: Optional[str] = None,
    max_size_bytes: Optional[int] = None,
) -> InputDocument:
    """Extracts text, page boundaries, and word positions from a PDF document.

    Requirements:
    - Extracts page-by-page preserving page boundary markers (`--- Page X ---`).
    - Preserves word coordinates for downstream spatial redactor / visual overlay.
    - Accurately flags OCR-required status for scanned (image-only) PDFs.
    - Distinguishes between scanned and genuinely empty PDFs.
    - Rejects malformed or corrupted PDFs with structured ExtractionError.
    - Strictly enforces file size limits.
    - Operates 100% on-device (no cloud processing).

    Args:
        file_bytes: Raw bytes or byte stream of the PDF.
        file_name: Optional original filename.
        max_size_bytes: Optional size override; defaults to MAX_FILE_SIZE_MB setting.

    Returns:
        InputDocument: Normalized document model with extracted text and layout metadata.

    Raises:
        ExtractionError: If the PDF is malformed, encrypted without password, empty, or exceeds size limits.
    """
    # 1. Normalize byte payload
    if file_bytes is None:
        raise ExtractionError("PDF input cannot be None.")

    if isinstance(file_bytes, (io.BytesIO, BinaryIO)):
        if hasattr(file_bytes, "seek"):
            file_bytes.seek(0)
        raw_bytes = file_bytes.read()
    elif isinstance(file_bytes, bytes):
        raw_bytes = file_bytes
    else:
        raise ExtractionError(f"Unsupported PDF payload type: {type(file_bytes).__name__}")

    size_bytes = len(raw_bytes)
    resolved_name = file_name or "document.pdf"

    # 2. Size limit enforcement
    if size_bytes == 0:
        raise ExtractionError("PDF file is empty (0 bytes).")

    settings = get_settings()
    effective_max_size = max_size_bytes or settings.max_file_size_bytes
    if size_bytes > effective_max_size:
        size_mb = size_bytes / (1024 * 1024)
        max_mb = effective_max_size / (1024 * 1024)
        raise ExtractionError(
            f"PDF file size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.2f} MB.",
            details={"file_name": resolved_name, "size_bytes": size_bytes, "max_bytes": effective_max_size},
        )

    # 3. Header signature check
    if not raw_bytes.startswith(b"%PDF-"):
        raise ExtractionError(
            f"File '{resolved_name}' does not have a valid PDF header signature (possible corruption or spoofing).",
            details={"file_name": resolved_name, "header_bytes": str(raw_bytes[:8])},
        )

    # 4. In-memory page-by-page extraction via pdfplumber
    page_texts: List[str] = []
    page_details: List[Dict[str, Any]] = []
    total_images = 0
    total_chars = 0
    total_words = 0

    try:
        stream = io.BytesIO(raw_bytes)
        with pdfplumber.open(stream) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise ExtractionError(
                    f"PDF document '{resolved_name}' contains zero pages.",
                    details={"file_name": resolved_name},
                )
            if page_count > 100:
                raise ExtractionError(
                    f"PDF document '{resolved_name}' exceeds maximum allowed threshold of 100 pages (contains {page_count} pages).",
                    details={"file_name": resolved_name, "page_count": page_count, "max_pages": 100},
                )

            for page_idx, page in enumerate(pdf.pages, start=1):
                # Extract text preserving formatting
                text = page.extract_text(layout=False) or ""
                text_clean = text.strip()
                char_count = len(text_clean)
                words_list = text_clean.split()
                word_count = len(words_list)

                # Extract word positions
                positions: List[Dict[str, Any]] = []
                try:
                    raw_words = page.extract_words() or []
                    for w in raw_words:
                        positions.append({
                            "text": w.get("text", ""),
                            "x0": round(float(w.get("x0", 0.0)), 2),
                            "top": round(float(w.get("top", 0.0)), 2),
                            "x1": round(float(w.get("x1", 0.0)), 2),
                            "bottom": round(float(w.get("bottom", 0.0)), 2),
                        })
                except Exception as pos_exc:
                    logger.debug("Positional word extraction omitted for page %d: %s", page_idx, str(pos_exc))

                image_count = len(page.images) if hasattr(page, "images") else 0
                total_images += image_count
                total_chars += char_count
                total_words += word_count

                page_details.append({
                    "page_number": page_idx,
                    "char_count": char_count,
                    "word_count": word_count,
                    "has_text": char_count > 0,
                    "image_count": image_count,
                    "has_images": image_count > 0,
                    "positions": positions,
                })

                # Preserve page boundaries in combined content
                page_texts.append(f"--- Page {page_idx} ---\n{text}")

    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(
            f"Failed to extract text from malformed or encrypted PDF '{resolved_name}': {exc}",
            details={"file_name": resolved_name, "error": str(exc)},
        ) from exc

    # 5. Scanned vs. Empty Detection & OCR-required determination
    combined_raw_text = "\n\n".join(page_texts)
    extracted_text_only = "".join(p.get("text", "") for p in page_details)

    # A PDF is considered scanned if text is absent or near-zero, but image elements exist
    is_scanned_pdf = (total_chars < (page_count * 15)) and (total_images > 0)
    is_empty_pdf = (total_chars == 0) and (total_images == 0)
    ocr_required = is_scanned_pdf

    metadata: Dict[str, Any] = {
        "source": "local_pdf_extractor",
        "mime_type": "application/pdf",
        "page_count": page_count,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_images": total_images,
        "is_scanned_pdf": is_scanned_pdf,
        "is_empty_pdf": is_empty_pdf,
        "ocr_required": ocr_required,
        "pages": page_details,
    }

    doc = InputDocument(
        document_type="PDF",
        raw_content=combined_raw_text if total_chars > 0 else "",
        raw_bytes=raw_bytes,
        file_name=resolved_name,
        file_size_bytes=size_bytes,
        metadata=metadata,
    )

    logger.info(
        "Extracted PDF [%s]: pages=%d, chars=%d, images=%d, scanned=%s, ocr_required=%s",
        resolved_name,
        page_count,
        total_chars,
        total_images,
        is_scanned_pdf,
        ocr_required,
    )

    return doc
