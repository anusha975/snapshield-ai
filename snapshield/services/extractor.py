"""Document and text extraction service for SnapShield AI.

Provides extraction capabilities across raw text strings, PDFs (via pdfplumber),
and images (via Pillow).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Tuple, Union
import pdfplumber
from PIL import Image

from snapshield.core.exceptions import ExtractionError
from snapshield.core.logger import logger
from snapshield.core.types import DocumentType


class DocumentExtractor:
    """Extracts raw text and metadata from diverse input sources."""

    @staticmethod
    def extract_from_text(text: str) -> Tuple[DocumentType, str]:
        """Validates and processes raw plain text input."""
        if not isinstance(text, str):
            raise ExtractionError("Input text must be a valid string.")
        return DocumentType.RAW_TEXT, text.strip()

    @staticmethod
    def extract_from_pdf(source: Union[str, Path, BinaryIO, bytes]) -> Tuple[DocumentType, str, int]:
        """Extracts text content and page count from a PDF document using pdfplumber."""
        try:
            stream = io.BytesIO(source) if isinstance(source, bytes) else source
            extracted_pages: list[str] = []

            with pdfplumber.open(stream) as pdf:
                page_count = len(pdf.pages)
                for idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_pages.append(page_text)

            combined_text = "\n\n".join(extracted_pages)
            logger.info("PDF extraction successful: extracted %d pages, %d characters.",
                        page_count, len(combined_text))
            return DocumentType.PDF, combined_text, page_count

        except Exception as exc:
            raise ExtractionError(f"Failed to extract text from PDF: {exc}") from exc

    @staticmethod
    def load_image(source: Union[str, Path, BinaryIO, bytes]) -> Tuple[DocumentType, Image.Image]:
        """Loads and verifies an image using Pillow for OCR processing."""
        try:
            stream = io.BytesIO(source) if isinstance(source, bytes) else source
            image = Image.open(stream)
            image.load()  # Verify image integrity
            return DocumentType.IMAGE, image
        except Exception as exc:
            raise ExtractionError(f"Failed to load image: {exc}") from exc
