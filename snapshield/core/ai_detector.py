"""Local AI contextual detection engine for SnapShield AI.

Features:
- Identifies contextual entities that deterministic regex rules cannot reliably extract:
  1. PERSON
  2. ADDRESS
  3. ORGANIZATION
  4. MEDICAL_INFORMATION
  5. PATIENT_IDENTIFIER
  6. CONFIDENTIAL_BUSINESS_INFORMATION
  7. PRIVATE_LOCATION
- Clean architectural abstractions:
  - AIDetector (Abstract Base Class)
  - LocalAIDetector (Concrete Implementation)
- Decoupled pipeline stages:
  1. Model loading separate from inference (load_model)
  2. Preprocessing separate from inference (preprocess)
  3. Inference execution (infer)
  4. Postprocessing separate from inference (postprocess)
- Confidence derived from actual model output probabilities (Softmax).
- Full model metadata tracking (runtime provider, architecture, fallback status).
- Guaranteed CPU fallback via SafeInferenceSession.
- Graceful model-unavailable state when on-device model is absent.
- Clearly distinguished development fallback that NEVER misrepresents fallback
  heuristics as real model inference.
- Zero external LLM or cloud API calls.
"""

from __future__ import annotations

import abc
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from snapshield.core.exceptions import DetectionError
from snapshield.core.logger import logger
from snapshield.core.models import Detection, SeverityLevel
from snapshield.npu.model_runner import ModelRunner
from snapshield.npu.runtime import RuntimeMode, get_active_provider


# =====================================================================
# Target Contextual Categories & Label Mappings
# =====================================================================

TARGET_CATEGORIES: List[str] = [
    "PERSON",
    "ADDRESS",
    "ORGANIZATION",
    "MEDICAL_INFORMATION",
    "PATIENT_IDENTIFIER",
    "CONFIDENTIAL_BUSINESS_INFORMATION",
    "PRIVATE_LOCATION",
]

# Mapping from external model NER/PII labels to standardized SnapShield categories
LABEL_CATEGORY_MAPPING: Dict[str, str] = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION": "PRIVATE_LOCATION",
    "COORDINATE": "PRIVATE_LOCATION",
    "PASSWORD": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "FINANCIAL": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "CREDIT_CARD": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "IBAN_CODE": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "US_BANK_NUMBER": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "MAC_ADDRESS": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "US_SSN": "PATIENT_IDENTIFIER",
    "US_PASSPORT": "PATIENT_IDENTIFIER",
    "US_DRIVER_LICENSE": "PATIENT_IDENTIFIER",
    "US_ITIN": "PATIENT_IDENTIFIER",
    "IMEI": "PATIENT_IDENTIFIER",
    "AGE": "PATIENT_IDENTIFIER",
    "TITLE": "PERSON",
    "ADDRESS": "ADDRESS",
    "MEDICAL_INFORMATION": "MEDICAL_INFORMATION",
    "CONFIDENTIAL_BUSINESS_INFORMATION": "CONFIDENTIAL_BUSINESS_INFORMATION",
    "PRIVATE_LOCATION": "PRIVATE_LOCATION",
}

# BIO Tag Mapping for Token Classification
# 0: O (Outside)
# 1..14: B- and I- tags for the 7 contextual categories
BIO_LABELS: List[str] = ["O"]
for cat in TARGET_CATEGORIES:
    BIO_LABELS.append(f"B-{cat}")
    BIO_LABELS.append(f"I-{cat}")

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(BIO_LABELS)}

# Category-specific severity mappings
CATEGORY_SEVERITY: Dict[str, SeverityLevel] = {
    "PATIENT_IDENTIFIER": SeverityLevel.CRITICAL,
    "MEDICAL_INFORMATION": SeverityLevel.HIGH,
    "CONFIDENTIAL_BUSINESS_INFORMATION": SeverityLevel.HIGH,
    "PRIVATE_LOCATION": SeverityLevel.HIGH,
    "ADDRESS": SeverityLevel.MEDIUM,
    "PERSON": SeverityLevel.MEDIUM,
    "ORGANIZATION": SeverityLevel.LOW,
}


# =====================================================================
# Domain Data Transfer Objects
# =====================================================================

@dataclass
class ModelMetadata:
    """Strongly typed model and runtime metadata."""
    model_name: str
    model_version: str
    model_path: Optional[str] = None
    execution_provider: str = "Uninitialized"
    is_loaded: bool = False
    is_fallback: bool = False
    context_window: int = 512
    target_categories: List[str] = field(default_factory=lambda: list(TARGET_CATEGORIES))
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "execution_provider": self.execution_provider,
            "is_loaded": self.is_loaded,
            "is_fallback": self.is_fallback,
            "context_window": self.context_window,
            "target_categories": self.target_categories,
            "description": self.description,
        }


@dataclass
class PreprocessedInput:
    """Tokenized and aligned representation prepared for inference."""
    original_text: str
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]
    token_offsets: List[Tuple[int, int]]  # (start_char, end_char) in original_text


@dataclass
class RawModelOutput:
    """Unprocessed output logits, probabilities, and execution diagnostics."""
    logits: np.ndarray  # Shape: [seq_len, num_classes]
    probabilities: np.ndarray  # Softmax probabilities [seq_len, num_classes]
    is_real_inference: bool
    execution_provider: str
    inference_time_ms: float


# =====================================================================
# Softmax Helper Function
# =====================================================================

def compute_softmax(logits: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax probabilities along the last axis."""
    if logits.ndim == 1:
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


# =====================================================================
# Abstract AI Detector Interface
# =====================================================================

class AIDetector(abc.ABC):
    """Abstract base contract for local on-device AI contextual detection."""

    @property
    @abc.abstractmethod
    def metadata(self) -> ModelMetadata:
        """Returns metadata regarding active model state, version, and provider."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Returns True if the detector is loaded and ready to process text."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, model_path: Optional[str | Path] = None) -> None:
        """Loads and initializes the model session separate from inference."""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, text: str) -> PreprocessedInput:
        """Transforms raw string into token features and offset maps separate from inference."""
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, model_input: PreprocessedInput) -> RawModelOutput:
        """Executes inference on preprocessed inputs separate from postprocessing."""
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(
        self,
        raw_output: RawModelOutput,
        preprocessed: PreprocessedInput,
        original_text: str,
    ) -> List[Detection]:
        """Transforms model probabilities and token offsets into Detection domain models."""
        raise NotImplementedError

    def detect(self, text: str) -> List[Detection]:
        """Orchestrates the detection pipeline: preprocess -> infer -> postprocess."""
        if not text or not isinstance(text, str):
            return []

        if not self.is_available:
            logger.info("Local AI detector is unavailable; returning 0 contextual detections.")
            return []

        preprocessed = self.preprocess(text)
        raw_output = self.infer(preprocessed)
        detections = self.postprocess(raw_output, preprocessed, text)
        return detections


# =====================================================================
# Concrete Local AI Detector Implementation
# =====================================================================

class LocalAIDetector(AIDetector):
    """Concrete local AI detector utilizing ONNX Runtime with CPU fallback.

    Architecture guarantees:
    - Never calls external LLMs or cloud endpoints.
    - True ONNX session execution when model is present.
    - Verified CPU fallback via SafeInferenceSession.
    - Clearly identified development fallback when model is absent.
    - Never fabricates or misrepresents fallback output as real model inference.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        allow_dev_fallback: bool = True,
        preferred_providers: Optional[List[str]] = None,
        context_window: int = 512,
        mode: Union[str, RuntimeMode] = RuntimeMode.AUTO,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._allow_dev_fallback = allow_dev_fallback
        self._preferred_providers = preferred_providers or ["CPUExecutionProvider"]
        self._context_window = context_window
        self._runtime_mode = mode if isinstance(mode, RuntimeMode) else RuntimeMode(mode.upper())

        self._runner: Optional[ModelRunner] = None
        self._session: Optional[Any] = None
        self._fast_tokenizer: Optional[Any] = None
        self._id2label: Optional[Dict[str, str]] = None
        self._is_available: bool = False
        self._metadata = ModelMetadata(
            model_name="snapshield-ner-contextual",
            model_version="1.0.0",
            model_path=str(self._model_path) if self._model_path else None,
            execution_provider="Uninitialized",
            is_loaded=False,
            is_fallback=False,
            context_window=self._context_window,
            target_categories=list(TARGET_CATEGORIES),
            description="On-device contextual token classification model for privacy checkpointing.",
        )

        # Separate model loading stage
        self.load_model(self._model_path)

    @property
    def metadata(self) -> ModelMetadata:
        """Returns active model metadata."""
        return self._metadata

    @property
    def is_available(self) -> bool:
        """Indicates whether real model or allowed dev fallback is active."""
        return self._is_available

    def load_model(self, model_path: Optional[str | Path] = None) -> None:
        """Loads and initializes the local ONNX model separate from inference."""
        resolved_path = Path(model_path) if model_path else self._model_path

        if resolved_path and resolved_path.exists() and resolved_path.is_file():
            try:
                from snapshield.npu.model_runner import ModelRunner

                runner = ModelRunner(
                    model_path=resolved_path,
                    mode=self._runtime_mode,
                )
                runner.initialize()

                self._runner = runner
                self._session = runner  # Keep backward-compatible attribute
                self._model_path = resolved_path
                self._is_available = True
                self._metadata.is_loaded = True
                self._metadata.is_fallback = False
                self._metadata.model_path = str(resolved_path)
                self._metadata.execution_provider = runner.active_provider
                self._metadata.description = (
                    f"Verified local ONNX session loaded via {runner.active_provider}. "
                    f"NPU Active: {runner.is_npu_active}, Accelerated: {runner.is_accelerated}."
                )

                # Attempt to load tokenizer.json and config.json if adjacent
                model_dir = resolved_path.parent
                cfg_path = model_dir / "config.json"
                tok_path = model_dir / "tokenizer.json"
                if cfg_path.exists():
                    try:
                        import json
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg_data = json.load(f)
                            self._id2label = cfg_data.get("id2label")
                    except Exception as cfg_err:
                        logger.warning("Could not parse config.json: %s", cfg_err)

                if tok_path.exists():
                    try:
                        import tokenizers
                        self._fast_tokenizer = tokenizers.Tokenizer.from_file(str(tok_path))
                    except Exception as tok_err:
                        logger.warning("Could not initialize fast tokenizer: %s", tok_err)

                logger.info(
                    "Local AI model successfully loaded via ModelRunner from %s (Provider: %s)",
                    resolved_path.name,
                    runner.active_provider,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load ONNX model at %s: %s. Transitioning to fallback state.",
                    resolved_path,
                    str(exc),
                )

        # Model is unavailable or failed to initialize
        if self._allow_dev_fallback:
            self._is_available = True
            self._metadata.is_loaded = False
            self._metadata.is_fallback = True
            self._metadata.model_path = str(resolved_path) if resolved_path else None
            self._metadata.execution_provider = "DevelopmentFallback"
            self._metadata.description = (
                "ONNX model file not found or uninitialized. Clearly marked development fallback active. "
                "Output is heuristic and must NEVER be represented as real model inference."
            )
            logger.info("Local AI detector running in clearly marked development fallback mode.")
        else:
            self._is_available = False
            self._metadata.is_loaded = False
            self._metadata.is_fallback = False
            self._metadata.execution_provider = "ModelUnavailable"
            self._metadata.description = "Model file unavailable; dev fallback disabled."
            logger.info("Local AI model unavailable state active (dev fallback disabled).")

    # =================================================================
    # Pipeline Stage 1: Preprocessing
    # =================================================================

    def preprocess(self, text: str) -> PreprocessedInput:
        """Tokenizes text and constructs token character offset alignments."""
        # Use fast subword tokenizer if loaded
        if self._fast_tokenizer is not None:
            try:
                enc = self._fast_tokenizer.encode(text)
                tokens = [text[s:e] if s < e else "" for s, e in enc.offsets]
                input_ids = enc.ids
                attention_mask = enc.attention_mask
                token_offsets = enc.offsets

                if len(tokens) > self._context_window:
                    tokens = tokens[: self._context_window]
                    token_offsets = token_offsets[: self._context_window]
                    input_ids = input_ids[: self._context_window]
                    attention_mask = attention_mask[: self._context_window]

                return PreprocessedInput(
                    original_text=text,
                    tokens=tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_offsets=token_offsets,
                )
            except Exception as tok_err:
                logger.debug("Fast tokenizer error, falling back to regex: %s", tok_err)

        tokens: List[str] = []
        token_offsets: List[Tuple[int, int]] = []

        # Word-level / punctuation tokenization with exact char offsets
        for match in re.finditer(r"\b[\w'-]+\b|[^\w\s]", text):
            token_str = match.group(0)
            tokens.append(token_str)
            token_offsets.append(match.span())

        # Truncate to context window
        if len(tokens) > self._context_window:
            tokens = tokens[: self._context_window]
            token_offsets = token_offsets[: self._context_window]

        # Generate deterministic input IDs (hash-based token ID mapping for vocab)
        input_ids = [abs(hash(t)) % 30522 for t in tokens]
        attention_mask = [1] * len(tokens)

        return PreprocessedInput(
            original_text=text,
            tokens=tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_offsets=token_offsets,
        )

    # =================================================================
    # Pipeline Stage 2: Inference
    # =================================================================

    def infer(self, model_input: PreprocessedInput) -> RawModelOutput:
        """Executes inference on preprocessed inputs separate from postprocessing."""
        start_time = time.perf_counter()
        seq_len = len(model_input.tokens)
        num_classes = len(BIO_LABELS)

        # Case A: Real ONNX inference via ModelRunner or SafeSession
        if (self._runner is not None or self._session is not None) and not self._metadata.is_fallback:
            try:
                feed: Dict[str, Any] = {
                    "input_ids": np.array([model_input.input_ids], dtype=np.int64),
                    "attention_mask": np.array([model_input.attention_mask], dtype=np.int64),
                }

                session_obj = (
                    getattr(self._runner, "session", None)
                    or getattr(self._session, "session", None)
                    or self._session
                )
                if session_obj and hasattr(session_obj, "get_inputs"):
                    input_names = [inp.name for inp in session_obj.get_inputs()]
                    if "token_type_ids" in input_names:
                        feed["token_type_ids"] = np.zeros_like(feed["input_ids"])

                if self._runner is not None and hasattr(self._runner, "run"):
                    outputs = self._runner.run(feed)
                    active_provider = self._runner.active_provider
                else:
                    outputs = self._session.run(None, feed)
                    active_provider = getattr(self._session, "verified_provider", "CPUExecutionProvider")

                logits_batch = outputs[0]  # Shape: [1, seq_len, num_classes] or [seq_len, num_classes]
                logits = np.squeeze(logits_batch, axis=0) if logits_batch.ndim == 3 else logits_batch
                probabilities = compute_softmax(logits)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                return RawModelOutput(
                    logits=logits,
                    probabilities=probabilities,
                    is_real_inference=True,
                    execution_provider=active_provider,
                    inference_time_ms=elapsed_ms,
                )
            except Exception as exc:
                logger.warning("ONNX inference failed: %s. Falling back to dev mode.", str(exc))
                # Degrade safely to fallback

        # Case B: Clearly marked development fallback
        # Generates synthetic logits based on contextual heuristics
        # RULE: NEVER represent fallback output as real model inference.
        logits = np.zeros((seq_len, num_classes), dtype=np.float32)
        logits[:, 0] = 3.0  # Default class 'O' high logit

        fallback_hits = self._run_contextual_heuristics(model_input.original_text)

        # Project heuristic hits onto token spans
        for cat_name, start_char, end_char in fallback_hits:
            b_label = f"B-{cat_name}"
            i_label = f"I-{cat_name}"
            b_id = LABEL_TO_ID.get(b_label, 0)
            i_id = LABEL_TO_ID.get(i_label, 0)

            is_first = True
            for idx, (t_start, t_end) in enumerate(model_input.token_offsets):
                # Check overlap
                if max(start_char, t_start) < min(end_char, t_end):
                    target_id = b_id if is_first else i_id
                    logits[idx, 0] = 0.5  # Lower 'O'
                    logits[idx, target_id] = 4.5  # Higher entity logit
                    is_first = False

        probabilities = compute_softmax(logits)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RawModelOutput(
            logits=logits,
            probabilities=probabilities,
            is_real_inference=False,
            execution_provider="DevelopmentFallback",
            inference_time_ms=elapsed_ms,
        )

    # =================================================================
    # Pipeline Stage 3: Postprocessing
    # =================================================================

    def postprocess(
        self,
        raw_output: RawModelOutput,
        preprocessed: PreprocessedInput,
        original_text: str,
    ) -> List[Detection]:
        """Transforms model probabilities and token offsets into Detection domain models."""
        if self._id2label and raw_output.is_real_inference:
            return self._postprocess_custom_labels(raw_output, preprocessed, original_text)

        detections: List[Detection] = []
        tokens = preprocessed.tokens
        offsets = preprocessed.token_offsets
        probabilities = raw_output.probabilities

        if len(tokens) == 0 or probabilities.shape[0] == 0:
            return []

        # Predict most probable class for each token
        predicted_classes = np.argmax(probabilities, axis=-1)

        current_category: Optional[str] = None
        span_start: Optional[int] = None
        span_end: Optional[int] = None
        span_confidences: List[float] = []

        def _flush_span() -> None:
            nonlocal current_category, span_start, span_end, span_confidences
            if current_category and span_start is not None and span_end is not None:
                # Confidence derived directly from actual model probability output
                avg_confidence = float(np.mean(span_confidences))
                raw_substr = original_text[span_start:span_end]

                severity = CATEGORY_SEVERITY.get(current_category, SeverityLevel.MEDIUM)

                if raw_output.is_real_inference:
                    detector_name = "local_ai_onnx"
                    explanation = (
                        f"Local contextual AI model detected {current_category} "
                        f"(confidence: {avg_confidence:.1%}, provider: {raw_output.execution_provider})."
                    )
                else:
                    detector_name = "local_ai_dev_fallback"
                    explanation = (
                        f"[DEV FALLBACK - NOT REAL INFERENCE] Contextual heuristic identified "
                        f"potential {current_category}."
                    )

                detections.append(
                    Detection(
                        entity_type=current_category,
                        start=span_start,
                        end=span_end,
                        confidence=round(avg_confidence, 4),
                        detector=detector_name,
                        valid=True,
                        severity=severity,
                        explanation=explanation,
                        raw_value=raw_substr,
                    )
                )

            current_category = None
            span_start = None
            span_end = None
            span_confidences = []

        for idx, cls_id in enumerate(predicted_classes):
            tag = ID_TO_LABEL.get(int(cls_id), "O")
            token_prob = float(probabilities[idx, cls_id])
            t_start, t_end = offsets[idx]

            if tag == "O":
                _flush_span()
            elif tag.startswith("B-"):
                _flush_span()
                current_category = tag[2:]
                span_start = t_start
                span_end = t_end
                span_confidences.append(token_prob)
            elif tag.startswith("I-"):
                tag_cat = tag[2:]
                if current_category == tag_cat:
                    span_end = t_end
                    span_confidences.append(token_prob)
                else:
                    _flush_span()
                    current_category = tag_cat
                    span_start = t_start
                    span_end = t_end
                    span_confidences.append(token_prob)

        _flush_span()

        # Sort detections by appearance
        detections.sort(key=lambda d: d.start)

        # Log results without leaking sensitive raw values
        logger.info(
            "Local AI postprocessing finished: extracted %d contextual entities (is_real_inference=%s)",
            len(detections),
            raw_output.is_real_inference,
        )

        return detections

    def _postprocess_custom_labels(
        self,
        raw_output: RawModelOutput,
        preprocessed: PreprocessedInput,
        original_text: str,
    ) -> List[Detection]:
        """Maps custom token classification classes (e.g. from bert-small-pii) to domain models."""
        detections: List[Detection] = []
        tokens = preprocessed.tokens
        offsets = preprocessed.token_offsets
        probabilities = raw_output.probabilities

        if len(tokens) == 0 or probabilities.shape[0] == 0 or not self._id2label:
            return []

        predicted_class_ids = np.argmax(probabilities, axis=-1)

        current_tag: Optional[str] = None
        span_start: Optional[int] = None
        span_end: Optional[int] = None
        span_confidences: List[float] = []

        def _flush() -> None:
            nonlocal current_tag, span_start, span_end, span_confidences
            if current_tag and span_start is not None and span_end is not None and span_start < span_end:
                mapped_cat = LABEL_CATEGORY_MAPPING.get(current_tag, current_tag)
                raw_slice = original_text[span_start:span_end].strip()
                if raw_slice and mapped_cat:
                    mean_conf = float(np.mean(span_confidences)) if span_confidences else 0.85
                    severity = CATEGORY_SEVERITY.get(mapped_cat, SeverityLevel.MEDIUM)

                    det = Detection(
                        entity_type=mapped_cat,
                        start=span_start,
                        end=span_end,
                        confidence=round(mean_conf, 4),
                        detector="local_ai_onnx",
                        valid=True,
                        severity=severity,
                        explanation=(
                            f"Detected contextual entity '{mapped_cat}' (label={current_tag}) "
                            f"via local ONNX model on {raw_output.execution_provider}."
                        ),
                        raw_value=raw_slice,
                    )
                    detections.append(det)

            current_tag = None
            span_start = None
            span_end = None
            span_confidences = []

        for idx, cls_id in enumerate(predicted_class_ids):
            label_name = self._id2label.get(str(cls_id), self._id2label.get(cls_id, "O"))
            prob = float(probabilities[idx, cls_id])

            if label_name == "O":
                _flush()
                continue

            tag = label_name[2:] if label_name.startswith(("B-", "I-")) else label_name
            t_start, t_end = offsets[idx]

            if label_name.startswith("B-"):
                _flush()
                current_tag = tag
                span_start = t_start
                span_end = t_end
                span_confidences.append(prob)
            elif label_name.startswith("I-"):
                if current_tag == tag and span_start is not None:
                    span_end = t_end
                    span_confidences.append(prob)
                else:
                    _flush()
                    current_tag = tag
                    span_start = t_start
                    span_end = t_end
                    span_confidences.append(prob)
            else:
                _flush()
                current_tag = tag
                span_start = t_start
                span_end = t_end
                span_confidences.append(prob)

        _flush()
        detections.sort(key=lambda d: d.start)
        return detections

    # =================================================================
    # Development Fallback Heuristics
    # =================================================================

    def _run_contextual_heuristics(self, text: str) -> List[Tuple[str, int, int]]:
        """Finds contextual entity spans for testing when model file is not present.

        Clearly designated as development fallback only.
        """
        hits: List[Tuple[str, int, int]] = []

        # 1. PATIENT_IDENTIFIER (Explicit clinical identifiers)
        for m in re.finditer(r"(?i)\b(?:MRN|Patient\s*(?:ID|#|Number)|Accession\s*#?|Health\s*ID)\s*[:=]?\s*([A-Za-z0-9\-]{5,20})\b", text):
            hits.append(("PATIENT_IDENTIFIER", m.start(), m.end()))

        # 2. PRIVATE_LOCATION (Preempts generic confidential words)
        for m in re.finditer(
            r"(?i)\b(?:private\s+residence|home\s+address|safe\s+house|personal\s+vacation\s+home|confidential\s+meeting\s+room|GPS\s+coordinates\s*:\s*-?\d+\.\d+,\s*-?\d+\.\d+)\b",
            text,
        ):
            hits.append(("PRIVATE_LOCATION", m.start(), m.end()))

        # 3. MEDICAL_INFORMATION
        for m in re.finditer(
            r"(?i)\b(?:diagnosed\s+with|prescription\s+for|dosage|chemotherapy|hypertension|insulin|biopsy\s+results|clinical\s+diagnosis|acute\s+symptoms|cardiac\s+arrest)\b",
            text,
        ):
            hits.append(("MEDICAL_INFORMATION", m.start(), m.end()))

        # 4. CONFIDENTIAL_BUSINESS_INFORMATION
        for m in re.finditer(
            r"(?i)\b(?:CONFIDENTIAL(?:\s+INFORMATION|\s+BUSINESS)?|PROJECT\s+[A-Z][A-Za-z0-9_]+|Q[1-4]\s+financial\s+forecast|M&A\s+(?:terms|negotiation)|internal\s+use\s+only|board\s+of\s+directors\s+memo)\b",
            text,
        ):
            hits.append(("CONFIDENTIAL_BUSINESS_INFORMATION", m.start(), m.end()))

        # 5. PERSON (Honorifics and name cues)
        for m in re.finditer(
            r"\b(?:Dr\.|Doctor|Mr\.|Mrs\.|Ms\.|Prof\.|Professor|Patient:|Physician:|Subject:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
            text,
        ):
            hits.append(("PERSON", m.start(), m.end()))

        # 6. ADDRESS (Street / Avenue / Road / Lane / Blvd / Postal)
        for m in re.finditer(
            r"\b\d{1,5}\s+[A-Z][A-Za-z0-9\s,\.]{2,40}(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Way)\b",
            text,
        ):
            hits.append(("ADDRESS", m.start(), m.end()))

        # 7. ORGANIZATION (Corporate & Institutional markers)
        for m in re.finditer(
            r"\b[A-Z][A-Za-z0-9&'\s]{1,35}(?:Inc\.|Corporation|Corp\.|LLC|Ltd\.|Hospital|Clinic|Technologies|University|Foundation|Bank)\b",
            text,
        ):
            # Avoid duplicate if already tagged as medical or address
            hits.append(("ORGANIZATION", m.start(), m.end()))

        # Deduplicate and sort spans
        deduped: List[Tuple[str, int, int]] = []
        for cat, start, end in hits:
            # Check overlap
            if not any(max(start, s) < min(end, e) for _, s, e in deduped):
                deduped.append((cat, start, end))

        deduped.sort(key=lambda x: x[1])
        return deduped


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bert_small_pii" / "model_quantized.onnx"


def create_real_ai_detector(
    model_path: Optional[str | Path] = None,
    mode: Union[str, RuntimeMode] = RuntimeMode.AUTO,
) -> LocalAIDetector:
    """Factory creating a LocalAIDetector wired to the real integrated ONNX model.

    Uses ModelRunner for inference, supporting CPU and verified accelerated providers.
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    return LocalAIDetector(
        model_path=target_path,
        allow_dev_fallback=False,
        mode=mode,
    )
