"""ONNX Runtime session manager with strict hardware verification and CPU fallback.

Enforces:
- Rule 4: Never fabricate Snapdragon/NPU performance.
- Rule 5: Never claim accelerated execution unless runtime verification confirms it.
- Rule 6: CPU fallback must always exist.
- Rule 14: Do not silently swallow exceptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import onnxruntime as ort

from snapshield.core.exceptions import HardwareAccelerationError
from snapshield.core.logger import logger
from snapshield.npu.device import probe_hardware_capabilities


class SafeInferenceSession:
    """Wraps an ONNX Runtime InferenceSession with verified provider fallback."""

    def __init__(
        self,
        model_path: str | Path,
        preferred_providers: Optional[List[str]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.preferred_providers = preferred_providers or ["CPUExecutionProvider"]
        self.sess_options = sess_options or ort.SessionOptions()
        self.session: Optional[ort.InferenceSession] = None
        self.verified_provider: str = "CPUExecutionProvider"

    def initialize(self) -> None:
        """Initializes the ONNX runtime session, falling back gracefully to CPU if acceleration fails."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at path: {self.model_path}")

        hardware_report = probe_hardware_capabilities()
        candidate_providers: List[str] = []

        for p in self.preferred_providers:
            if p in hardware_report.available_onnx_providers:
                candidate_providers.append(p)

        # Always ensure CPUExecutionProvider is available at the end of the candidate list
        if "CPUExecutionProvider" not in candidate_providers:
            candidate_providers.append("CPUExecutionProvider")

        try:
            logger.info("Initializing ONNX session for %s with candidate providers: %s",
                        self.model_path.name, candidate_providers)
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=self.sess_options,
                providers=candidate_providers,
            )
            # Query runtime-verified active providers from the session itself
            actual_providers = self.session.get_providers()
            self.verified_provider = actual_providers[0] if actual_providers else "CPUExecutionProvider"
            logger.info("ONNX session successfully established. Verified active provider: %s", self.verified_provider)

        except Exception as exc:
            # If accelerated provider failed, attempt pure CPU fallback
            logger.warning("Primary provider initialization failed (%s). Attempting explicit CPU fallback.", str(exc))
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=self.sess_options,
                    providers=["CPUExecutionProvider"],
                )
                self.verified_provider = "CPUExecutionProvider"
                logger.info("Pure CPU fallback successfully initialized.")
            except Exception as cpu_exc:
                raise HardwareAccelerationError(
                    f"Failed to initialize ONNX session even with CPU fallback: {cpu_exc}",
                    details={"model": str(self.model_path), "original_error": str(exc), "fallback_error": str(cpu_exc)}
                ) from cpu_exc

    def run(self, output_names: Optional[List[str]], input_feed: dict[str, Any]) -> list[Any]:
        """Runs inference on the verified session."""
        if self.session is None:
            raise RuntimeError("Inference session has not been initialized. Call initialize() first.")
        return self.session.run(output_names, input_feed)
