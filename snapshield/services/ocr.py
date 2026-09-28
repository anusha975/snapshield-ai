"""Local Optical Character Recognition (OCR) Service Interface.

Provides optical text extraction from screenshots and images, with graceful fallback
and explicit error reporting.
"""

from __future__ import annotations

from typing import Tuple
from PIL import Image

from snapshield.core.exceptions import OCRError
from snapshield.core.logger import logger


class LocalOCRService:
    """Service for running local on-device OCR on screenshots and scanned documents."""

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold
        self._engine_available = self._check_engine_availability()

    def _check_engine_availability(self) -> bool:
        """Checks if local OCR engine (e.g. pytesseract or Windows native OCR) is installed."""
        try:
            import pytesseract
            # Test if tesseract binary is accessible
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.info("Local tesseract OCR binary not detected in PATH; OCR fallback mode enabled.")
            return False

    def is_available(self) -> bool:
        """Returns True if local OCR engine is ready for processing."""
        return self._engine_available

    def extract_text_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """Extracts text from a PIL Image and returns extracted string and verified confidence score.

        Enforces Rule 3: Never fabricate AI confidence.
        """
        if not self._engine_available:
            # Report truthful status instead of faking OCR results or confidence
            logger.warning("OCR requested but no local OCR engine is available.")
            raise OCRError(
                "Local OCR engine is not installed or available on this system. "
                "Install Tesseract OCR or configure the Windows Media OCR bridge."
            )

        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            extracted_words = []
            confidences = []

            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                if word.strip() and int(conf) > 0:
                    extracted_words.append(word)
                    confidences.append(float(conf) / 100.0)

            full_text = " ".join(extracted_words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return full_text, avg_confidence

        except Exception as exc:
            raise OCRError(f"OCR execution failed: {exc}") from exc
