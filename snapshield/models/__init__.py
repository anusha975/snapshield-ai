"""Local Models Module for SnapShield AI.

Provides model directory resolution and verification helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).resolve().parent
BERT_SMALL_PII_DIR = MODELS_DIR / "bert_small_pii"
DEFAULT_MODEL_FILE = BERT_SMALL_PII_DIR / "model_quantized.onnx"
DEFAULT_CONFIG_FILE = BERT_SMALL_PII_DIR / "config.json"
DEFAULT_TOKENIZER_FILE = BERT_SMALL_PII_DIR / "tokenizer.json"


def get_default_model_path() -> Optional[Path]:
    """Returns path to the default local ONNX model file if installed on disk."""
    if DEFAULT_MODEL_FILE.exists() and DEFAULT_MODEL_FILE.is_file():
        return DEFAULT_MODEL_FILE
    return None


def is_default_model_installed() -> bool:
    """Returns True if the default quantized ONNX model and metadata are present."""
    return (
        DEFAULT_MODEL_FILE.exists()
        and DEFAULT_CONFIG_FILE.exists()
        and DEFAULT_TOKENIZER_FILE.exists()
    )


__all__ = [
    "BERT_SMALL_PII_DIR",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_MODEL_FILE",
    "DEFAULT_TOKENIZER_FILE",
    "MODELS_DIR",
    "get_default_model_path",
    "is_default_model_installed",
]
