"""ONNX Model Runner with Hardware-Aware Provider Selection and CPU Fallback.

Enforces:
- Rule 4: Never fabricate Snapdragon/NPU performance.
- Rule 5: Never claim accelerated execution unless runtime verification confirms it.
- Rule 6: CPU fallback must always exist.
- Never claim "NPU active" unless the active provider is confirmed by ONNX Runtime.
- Mode support: AUTO, ACCELERATED, CPU.
- Graceful session management and latency benchmarking.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ORT_AVAILABLE = False

from snapshield.core.exceptions import HardwareAccelerationError
from snapshield.core.logger import logger
from snapshield.core.models import BenchmarkResult
from snapshield.npu.runtime import (
    ACCELERATED_PROVIDERS,
    NPU_PROVIDERS,
    RuntimeMode,
    get_active_provider,
    get_available_providers,
)


# Global thread-safe session cache: (model_path_str, tuple(providers)) -> ort.InferenceSession
_GLOBAL_SESSION_CACHE: Dict[Tuple[str, Tuple[str, ...]], Any] = {}


def clear_model_session_cache() -> None:
    """Clears the cached ONNX model inference sessions."""
    global _GLOBAL_SESSION_CACHE
    _GLOBAL_SESSION_CACHE.clear()


def get_model_session_cache_size() -> int:
    """Returns the number of cached ONNX inference sessions."""
    return len(_GLOBAL_SESSION_CACHE)



class ModelRunner:
    """Manages ONNX model inference sessions with factual hardware provider selection."""

    def __init__(
        self,
        model_path: Union[str, Path],
        mode: Union[str, RuntimeMode] = RuntimeMode.AUTO,
        session_options: Optional[Any] = None,
        provider_options: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.mode = mode if isinstance(mode, RuntimeMode) else RuntimeMode(mode.upper())
        self.session_options = session_options
        self.provider_options = provider_options
        self.use_cache = use_cache

        self.session: Optional[Any] = None
        self.active_provider: str = "Uninitialized"
        self.is_npu_active: bool = False
        self.is_accelerated: bool = False
        self.fallback_occurred: bool = False
        self.initialization_error: Optional[str] = None

    def initialize(self) -> None:
        """Initializes the ONNX runtime inference session with verified provider fallback."""
        if not ORT_AVAILABLE:
            raise HardwareAccelerationError(
                "ONNX Runtime is not installed. Model inference cannot be initialized.",
                details={"model": str(self.model_path)},
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Build candidate providers based on mode
        candidate_providers = self._resolve_candidate_providers()

        try:
            logger.info(
                "Initializing ModelRunner for '%s' (mode=%s) with candidate providers: %s",
                self.model_path.name,
                self.mode.value,
                candidate_providers,
            )
            self._create_session(candidate_providers)

        except Exception as exc:
            # If accelerated provider fails, attempt explicit CPU fallback
            logger.warning(
                "ModelRunner failed initialization with primary candidate providers (%s): %s. "
                "Attempting guaranteed CPU fallback.",
                candidate_providers,
                exc,
            )
            self.fallback_occurred = True
            self.initialization_error = str(exc)

            try:
                self._create_session(["CPUExecutionProvider"])
                logger.info("ModelRunner successfully initialized with CPUExecutionProvider fallback.")
            except Exception as cpu_exc:
                raise HardwareAccelerationError(
                    f"Failed to initialize ModelRunner even with CPU fallback: {cpu_exc}",
                    details={
                        "model": str(self.model_path),
                        "original_error": str(exc),
                        "fallback_error": str(cpu_exc),
                    },
                ) from cpu_exc

        # Verify actual provider selected by the ONNX Runtime session
        actual_providers = self.session.get_providers()
        self.active_provider = actual_providers[0] if actual_providers else "CPUExecutionProvider"

        # Truthfulness invariant: NEVER claim NPU active unless verified
        self.is_npu_active = (self.active_provider in NPU_PROVIDERS)
        self.is_accelerated = (
            self.active_provider in ACCELERATED_PROVIDERS
            and self.active_provider != "CPUExecutionProvider"
        )

        logger.info(
            "ModelRunner initialized for '%s'. Active provider: %s (NPU Active: %s, Accelerated: %s)",
            self.model_path.name,
            self.active_provider,
            self.is_npu_active,
            self.is_accelerated,
        )

    def _resolve_candidate_providers(self) -> List[str]:
        """Resolves candidate providers according to mode and local environment."""
        if self.mode == RuntimeMode.CPU:
            return ["CPUExecutionProvider"]

        available = get_available_providers()
        candidates: List[str] = []

        if self.mode in (RuntimeMode.AUTO, RuntimeMode.ACCELERATED):
            # Prioritize QNN if present in available providers
            if "QNNExecutionProvider" in available:
                candidates.append("QNNExecutionProvider")

            # Add other available accelerators
            for p in ACCELERATED_PROVIDERS:
                if p != "QNNExecutionProvider" and p in available and p not in candidates:
                    candidates.append(p)

        # Always add CPUExecutionProvider at the end for runtime fallback
        if "CPUExecutionProvider" not in candidates:
            candidates.append("CPUExecutionProvider")

        return candidates

    def _create_session(self, providers: List[str]) -> None:
        """Internal helper to instantiate or retrieve cached InferenceSession."""
        cache_key = (str(self.model_path.resolve()), tuple(providers))

        if self.use_cache and cache_key in _GLOBAL_SESSION_CACHE:
            self.session = _GLOBAL_SESSION_CACHE[cache_key]
            logger.debug("Reusing cached ONNX InferenceSession for %s (%s)", self.model_path.name, providers)
            return

        sess_opts = self.session_options or ort.SessionOptions()
        if self.provider_options and len(self.provider_options) == len(providers):
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=providers,
                provider_options=self.provider_options,
            )
        else:
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=providers,
            )

        self.session = session
        if self.use_cache:
            _GLOBAL_SESSION_CACHE[cache_key] = session

    def run(
        self,
        input_feed: Dict[str, Any],
        output_names: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Executes inference on the initialized session."""
        if self.session is None:
            self.initialize()
        return self.session.run(output_names, input_feed)

    def benchmark(
        self,
        input_feed: Dict[str, Any],
        iterations: int = 10,
        warmup: int = 2,
    ) -> BenchmarkResult:
        """Runs a performance benchmark measuring latency factually."""
        if self.session is None:
            self.initialize()

        if iterations < 1:
            raise ValueError(f"Iterations must be at least 1, got {iterations}")

        # Warmup passes
        for _ in range(max(0, warmup)):
            self.session.run(None, input_feed)

        # Timed passes
        latencies_ms: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.session.run(None, input_feed)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

        return BenchmarkResult(
            execution_provider=self.active_provider,
            sample_name=self.model_path.name,
            iterations=iterations,
            mean_latency_ms=round(mean_lat, 3),
            p95_latency_ms=round(p95_lat, 3),
            throughput_items_per_sec=round(throughput, 2),
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns session inputs, outputs, and factual hardware status."""
        if self.session is None:
            return {
                "model_path": str(self.model_path),
                "initialized": False,
                "active_provider": self.active_provider,
                "is_npu_active": False,
                "is_accelerated": False,
            }

        inputs_info = [
            {"name": node.name, "type": node.type, "shape": node.shape}
            for node in self.session.get_inputs()
        ]
        outputs_info = [
            {"name": node.name, "type": node.type, "shape": node.shape}
            for node in self.session.get_outputs()
        ]

        return {
            "model_path": str(self.model_path),
            "initialized": True,
            "mode": self.mode.value,
            "active_provider": self.active_provider,
            "is_npu_active": self.is_npu_active,
            "is_accelerated": self.is_accelerated,
            "fallback_occurred": self.fallback_occurred,
            "inputs": inputs_info,
            "outputs": outputs_info,
        }

    def close(self) -> None:
        """Releases session resources."""
        self.session = None

    def __enter__(self) -> ModelRunner:
        if self.session is None:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
