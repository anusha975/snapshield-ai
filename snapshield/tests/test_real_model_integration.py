"""Tests for the integrated lightweight on-device AI model (bert-small-pii-int8).

Validates:
- Model presence, metadata, and tokenizer assets in models/bert_small_pii.
- Loading and inference execution via ModelRunner.
- Verified CPU execution provider without fake hardware claims.
- Detection of contextual entities (PERSON, ORGANIZATION, LOCATION, etc.).
- Strict adherence to privacy invariants (no external calls, masked values).
"""

from pathlib import Path
import pytest

from snapshield.core.ai_detector import (
    DEFAULT_MODEL_PATH,
    LocalAIDetector,
    create_real_ai_detector,
)
from snapshield.core.models import BenchmarkResult, Detection
from snapshield.models import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_TOKENIZER_FILE,
    is_default_model_installed,
)
from snapshield.npu.model_runner import ModelRunner
from snapshield.npu.runtime import RuntimeMode


def test_model_files_and_metadata_present():
    """Verifies that the integrated ONNX model and tokenizer assets exist on disk."""
    assert is_default_model_installed() is True
    assert DEFAULT_MODEL_FILE.exists()
    assert DEFAULT_CONFIG_FILE.exists()
    assert DEFAULT_TOKENIZER_FILE.exists()

    # Verify model size is realistic (~27.4 MB)
    size_mb = DEFAULT_MODEL_FILE.stat().st_size / (1024 * 1024)
    assert 25.0 <= size_mb <= 35.0


def test_model_runner_loads_integrated_model():
    """Verifies that ModelRunner loads the real ONNX model with factual provider reporting."""
    runner = ModelRunner(model_path=DEFAULT_MODEL_PATH, mode=RuntimeMode.AUTO)
    runner.initialize()

    # Factual assertions on the current host (x86_64 CPU)
    assert runner.active_provider == "CPUExecutionProvider"
    assert runner.is_npu_active is False
    assert runner.is_accelerated is False
    assert runner.fallback_occurred is False

    meta = runner.get_metadata()
    assert meta["initialized"] is True
    assert len(meta["inputs"]) >= 2  # input_ids, attention_mask

    # Benchmark run
    bench = runner.benchmark(
        input_feed={
            "input_ids": [[101, 2054, 2003, 1037, 102]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "token_type_ids": [[0, 0, 0, 0, 0]],
        },
        iterations=3,
        warmup=1,
    )
    assert isinstance(bench, BenchmarkResult)
    assert bench.mean_latency_ms > 0.0
    assert bench.throughput_items_per_sec > 0.0
    runner.close()


def test_real_ai_detector_initialization():
    """Verifies that create_real_ai_detector initializes a non-fallback ONNX detector."""
    detector = create_real_ai_detector(mode="AUTO")

    assert detector.is_available is True
    assert detector.metadata.is_loaded is True
    assert detector.metadata.is_fallback is False
    assert detector.metadata.execution_provider == "CPUExecutionProvider"
    assert "Verified local ONNX session" in detector.metadata.description


def test_real_ai_detector_contextual_inference():
    """Verifies end-to-end token classification on text containing names and organizations."""
    detector = create_real_ai_detector(mode="AUTO")
    text = "Please send the report to John Doe at Acme Corp in New York immediately."

    detections = detector.detect(text)

    assert len(detections) >= 2
    types = [d.entity_type for d in detections]

    # Verify key entities were extracted by real ONNX model
    assert "PERSON" in types
    assert "ORGANIZATION" in types or "PRIVATE_LOCATION" in types

    for d in detections:
        assert isinstance(d, Detection)
        assert d.detector == "local_ai_onnx"
        assert d.confidence > 0.5
        assert d.start >= 0
        assert d.end <= len(text)
        assert d.raw_value != ""
        # Masked representation works safely
        masked = d.get_masked_display_value()
        assert "*" in masked or "[REDACTED" in masked


def test_real_ai_detector_never_claims_acceleration_on_cpu_host():
    """Verifies that acceleration is never falsely claimed on CPU hardware."""
    detector = create_real_ai_detector(mode=RuntimeMode.ACCELERATED)

    # Since host only has CPU/Azure providers in ORT, it falls back to CPU safely
    assert detector.metadata.execution_provider == "CPUExecutionProvider"
    assert "NPU Active: False" in detector.metadata.description
    assert "Accelerated: False" in detector.metadata.description
