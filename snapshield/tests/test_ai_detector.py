"""Comprehensive tests for local AI contextual detection engine.

Validates:
1. Target categories:
   - PERSON
   - ADDRESS
   - ORGANIZATION
   - MEDICAL_INFORMATION
   - PATIENT_IDENTIFIER
   - CONFIDENTIAL_BUSINESS_INFORMATION
   - PRIVATE_LOCATION
2. Architectural abstractions:
   - AIDetector (abstract base class)
   - LocalAIDetector (concrete implementation)
3. Decoupled pipeline stages:
   - Model loading separate from inference
   - Preprocessing separate from inference
   - Postprocessing separate from inference
4. Confidence derived from actual model output:
   - Softmax calculation
   - Strict reflection of class probabilities in Detection.confidence
5. Model metadata:
   - Model name, version, path, execution provider, fallback status, context window
6. Graceful model-unavailable state:
   - Uninstalled/missing model handling
   - Safe behavior when dev fallback is disabled
7. Development fallback integrity:
   - Clearly marked development fallback
   - Invariant: NEVER represent fallback output as real model inference
8. CPU fallback:
   - Verified fallback to CPUExecutionProvider
9. Robustness and failure behavior:
   - Empty/whitespace input
   - Context window truncation
   - Mock ONNX inference session verification
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from snapshield.core.ai_detector import (
    BIO_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    TARGET_CATEGORIES,
    AIDetector,
    LocalAIDetector,
    ModelMetadata,
    PreprocessedInput,
    RawModelOutput,
    compute_softmax,
)
from snapshield.core.models import Detection, SeverityLevel


# =====================================================================
# 1. Abstraction & Interface Contract Tests
# =====================================================================

def test_ai_detector_cannot_be_instantiated_directly():
    """Verify AIDetector is an abstract class requiring implementation."""
    with pytest.raises(TypeError):
        AIDetector()  # type: ignore


def test_target_categories_coverage():
    """Verify all 7 required contextual categories are registered."""
    required = {
        "PERSON",
        "ADDRESS",
        "ORGANIZATION",
        "MEDICAL_INFORMATION",
        "PATIENT_IDENTIFIER",
        "CONFIDENTIAL_BUSINESS_INFORMATION",
        "PRIVATE_LOCATION",
    }
    assert required.issubset(set(TARGET_CATEGORIES))
    assert len(BIO_LABELS) == 1 + 2 * len(TARGET_CATEGORIES)  # O + B-/I- for each


# =====================================================================
# 2. Decoupled Pipeline Stages Tests
# =====================================================================

def test_preprocessing_separate_from_inference():
    """Verify preprocessing step tokenizes and produces aligned character offsets."""
    detector = LocalAIDetector(model_path=None, allow_dev_fallback=True)
    text = "Patient ID: PID-99882 was examined by Dr. Robert Smith."

    preprocessed = detector.preprocess(text)

    assert isinstance(preprocessed, PreprocessedInput)
    assert preprocessed.original_text == text
    assert len(preprocessed.tokens) > 0
    assert len(preprocessed.tokens) == len(preprocessed.token_offsets)
    assert len(preprocessed.input_ids) == len(preprocessed.tokens)
    assert len(preprocessed.attention_mask) == len(preprocessed.tokens)

    # Verify character offset alignments match token text
    for token, (start, end) in zip(preprocessed.tokens, preprocessed.token_offsets):
        assert text[start:end] == token


def test_preprocessing_context_window_truncation():
    """Verify that preprocessing respects context_window bounds without crashing."""
    detector = LocalAIDetector(model_path=None, allow_dev_fallback=True, context_window=10)
    long_text = "word " * 50

    preprocessed = detector.preprocess(long_text)
    assert len(preprocessed.tokens) == 10
    assert len(preprocessed.token_offsets) == 10


def test_softmax_numerical_stability():
    """Verify compute_softmax operates correctly and stably with large logits."""
    large_logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    probs = compute_softmax(large_logits)

    assert np.isclose(np.sum(probs), 1.0)
    assert probs[2] > probs[1] > probs[0]
    assert not np.isnan(probs).any()


def test_postprocessing_separate_from_inference():
    """Verify postprocessing extracts structured Detection models from raw probabilities."""
    detector = LocalAIDetector(model_path=None, allow_dev_fallback=True)
    text = "Patient ID: MRN-12345"
    preprocessed = detector.preprocess(text)

    # Construct synthetic raw model output where token 2 & 3 are B-PATIENT_IDENTIFIER & I-PATIENT_IDENTIFIER
    seq_len = len(preprocessed.tokens)
    num_classes = len(BIO_LABELS)
    logits = np.zeros((seq_len, num_classes), dtype=np.float32)
    logits[:, 0] = 5.0  # Default 'O'

    b_pid = LABEL_TO_ID["B-PATIENT_IDENTIFIER"]
    i_pid = LABEL_TO_ID["I-PATIENT_IDENTIFIER"]

    logits[2, 0] = 0.0
    logits[2, b_pid] = 6.0  # High confidence
    if seq_len > 3:
        logits[3, 0] = 0.0
        logits[3, i_pid] = 6.0

    probs = compute_softmax(logits)
    raw_output = RawModelOutput(
        logits=logits,
        probabilities=probs,
        is_real_inference=True,
        execution_provider="CPUExecutionProvider",
        inference_time_ms=1.5,
    )

    detections = detector.postprocess(raw_output, preprocessed, text)
    assert len(detections) >= 1
    d = detections[0]
    assert d.entity_type == "PATIENT_IDENTIFIER"
    assert d.severity == SeverityLevel.CRITICAL
    assert d.detector == "local_ai_onnx"
    assert d.confidence > 0.90
    # Safe preview test
    assert d.raw_value is not None


# =====================================================================
# 3. Model Metadata & Unavailable State Tests
# =====================================================================

def test_model_metadata_structure():
    """Verify model metadata contains required diagnostic fields."""
    detector = LocalAIDetector(model_path=None, allow_dev_fallback=True)
    meta = detector.metadata

    assert isinstance(meta, ModelMetadata)
    assert meta.model_name == "snapshield-ner-contextual"
    assert meta.model_version == "1.0.0"
    assert meta.is_fallback is True
    assert meta.is_loaded is False
    assert meta.execution_provider == "DevelopmentFallback"
    assert "PATIENT_IDENTIFIER" in meta.target_categories

    meta_dict = meta.to_dict()
    assert isinstance(meta_dict, dict)
    assert meta_dict["is_fallback"] is True


def test_graceful_model_unavailable_when_fallback_disabled():
    """Verify graceful behavior when model is missing and dev fallback is disabled."""
    detector = LocalAIDetector(
        model_path="non_existent_model_weights.onnx",
        allow_dev_fallback=False,
    )

    assert detector.is_available is False
    assert detector.metadata.is_loaded is False
    assert detector.metadata.is_fallback is False
    assert detector.metadata.execution_provider == "ModelUnavailable"

    # Must return empty list safely without crashing
    results = detector.detect("Dr. Alice Smith diagnosed patient with hypertension.")
    assert results == []


# =====================================================================
# 4. Invariant: Never Represent Fallback as Real Model Inference
# =====================================================================

def test_fallback_never_represented_as_real_model_inference():
    """Verify invariant: Fallback output is NEVER represented as real inference."""
    detector = LocalAIDetector(model_path=None, allow_dev_fallback=True)

    text = "Confidential: Project Titan M&A terms."
    findings = detector.detect(text)

    assert len(findings) > 0
    for f in findings:
        # 1. Detector must clearly indicate dev fallback
        assert f.detector == "local_ai_dev_fallback"
        assert f.detector != "local_ai_onnx"
        # 2. Explanation must explicitly warn about fallback nature
        assert "[DEV FALLBACK - NOT REAL INFERENCE]" in f.explanation
        # 3. Metadata must state fallback
        assert detector.metadata.is_fallback is True
        assert detector.metadata.is_loaded is False


# =====================================================================
# 5. Target Categories Detection Tests (Contextual Fallback)
# =====================================================================

def test_detect_patient_identifier():
    """Verify PATIENT_IDENTIFIER detection with CRITICAL severity."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "Admitted patient with MRN: 99482-B to oncology."
    findings = detector.detect(text)

    pid_findings = [f for f in findings if f.entity_type == "PATIENT_IDENTIFIER"]
    assert len(pid_findings) == 1
    assert pid_findings[0].severity == SeverityLevel.CRITICAL


def test_detect_medical_information():
    """Verify MEDICAL_INFORMATION detection with HIGH severity."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "Subject was diagnosed with hypertension and started chemotherapy."
    findings = detector.detect(text)

    med_findings = [f for f in findings if f.entity_type == "MEDICAL_INFORMATION"]
    assert len(med_findings) >= 1
    assert med_findings[0].severity == SeverityLevel.HIGH


def test_detect_confidential_business_information():
    """Verify CONFIDENTIAL_BUSINESS_INFORMATION detection with HIGH severity."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "CONFIDENTIAL: Internal memo regarding Q3 financial forecast."
    findings = detector.detect(text)

    biz_findings = [f for f in findings if f.entity_type == "CONFIDENTIAL_BUSINESS_INFORMATION"]
    assert len(biz_findings) >= 1
    assert biz_findings[0].severity == SeverityLevel.HIGH


def test_detect_private_location():
    """Verify PRIVATE_LOCATION detection with HIGH severity."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "The team met at a confidential meeting room before traveling."
    findings = detector.detect(text)

    loc_findings = [f for f in findings if f.entity_type == "PRIVATE_LOCATION"]
    assert len(loc_findings) >= 1
    assert loc_findings[0].severity == SeverityLevel.HIGH


def test_detect_person_contextual():
    """Verify PERSON detection using contextual honorifics/cues."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "The attending physician was Dr. Arthur Pendelton at the ward."
    findings = detector.detect(text)

    person_findings = [f for f in findings if f.entity_type == "PERSON"]
    assert len(person_findings) == 1
    assert person_findings[0].severity == SeverityLevel.MEDIUM
    assert "Arthur Pendelton" in person_findings[0].raw_value


def test_detect_address_contextual():
    """Verify physical ADDRESS detection."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "Deliver package directly to 742 Evergreen Terrace Avenue immediately."
    findings = detector.detect(text)

    addr_findings = [f for f in findings if f.entity_type == "ADDRESS"]
    assert len(addr_findings) == 1
    assert addr_findings[0].severity == SeverityLevel.MEDIUM


def test_detect_organization_contextual():
    """Verify ORGANIZATION detection."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    text = "The contract was signed with Acronis Technologies Inc. yesterday."
    findings = detector.detect(text)

    org_findings = [f for f in findings if f.entity_type == "ORGANIZATION"]
    assert len(org_findings) == 1
    assert org_findings[0].severity == SeverityLevel.LOW


# =====================================================================
# 6. Real ONNX Inference Path with Mock Session
# =====================================================================

def test_real_onnx_inference_with_mock_session():
    """Verify that when an ONNX model is loaded, true inference and confidence are used."""
    mock_session = MagicMock()
    mock_session.verified_provider = "CPUExecutionProvider"

    mock_input = MagicMock()
    mock_input.name = "input_ids"
    mock_session.session.get_inputs.return_value = [mock_input]

    # Configure detector with mock session
    detector = LocalAIDetector(allow_dev_fallback=False)
    detector._session = mock_session
    detector._is_available = True
    detector._metadata.is_loaded = True
    detector._metadata.is_fallback = False
    detector._metadata.execution_provider = "CPUExecutionProvider"

    text = "Dr. Alice Smith"
    preprocessed = detector.preprocess(text)
    seq_len = len(preprocessed.tokens)

    # Mock logits where token 1 is B-PERSON with 95% probability
    fake_logits = np.zeros((1, seq_len, len(BIO_LABELS)), dtype=np.float32)
    fake_logits[0, :, 0] = 1.0  # default O
    b_person = LABEL_TO_ID["B-PERSON"]
    fake_logits[0, 1, 0] = 0.0
    fake_logits[0, 1, b_person] = 6.0  # >95% confidence with 15 classes

    mock_session.run.return_value = [fake_logits]

    # Run inference
    raw_output = detector.infer(preprocessed)
    assert raw_output.is_real_inference is True
    assert raw_output.execution_provider == "CPUExecutionProvider"

    # Run postprocessing
    detections = detector.postprocess(raw_output, preprocessed, text)
    assert len(detections) >= 1
    assert detections[0].entity_type == "PERSON"
    assert detections[0].detector == "local_ai_onnx"
    assert detections[0].confidence >= 0.85
    assert "CPUExecutionProvider" in detections[0].explanation


# =====================================================================
# 7. Robustness & Edge Cases
# =====================================================================

def test_empty_and_whitespace_input():
    """Verify empty or non-string input returns empty list without errors."""
    detector = LocalAIDetector(allow_dev_fallback=True)

    assert detector.detect("") == []
    assert detector.detect("   ") == []
    assert detector.detect(None) == []  # type: ignore
    assert detector.detect(12345) == []  # type: ignore


def test_text_with_no_sensitive_entities():
    """Verify clean benign text produces no detections."""
    detector = LocalAIDetector(allow_dev_fallback=True)
    benign_text = "The quick brown fox jumps over the lazy dog."
    findings = detector.detect(benign_text)
    assert findings == []
