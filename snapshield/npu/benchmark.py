"""Hardware and Execution Provider Benchmarking Suite for SnapShield AI.

Benchmarks identical on-device AI model workloads across:
1. CPU (CPUExecutionProvider)
2. Verified accelerated provider (QNNExecutionProvider, DmlExecutionProvider, etc.), if available.

Measures:
- First inference (cold latency)
- Warm inference (first post-warmup latency)
- Mean latency
- Median latency
- Min latency
- Max latency
- Standard deviation
- Throughput (items/sec)

Strict Invariants:
- Never invents or fabricates benchmark metrics.
- Warm up before steady-state measurements.
- If acceleration is unavailable: displays 'Accelerated execution: Not available'.
- Never displays fake comparison values.
- Returns strongly-typed BenchmarkResult objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from snapshield.core.exceptions import HardwareAccelerationError
from snapshield.core.logger import logger
from snapshield.core.models import BenchmarkResult
from snapshield.models import (
    DEFAULT_MODEL_FILE,
    DEFAULT_TOKENIZER_FILE,
    get_default_model_path,
)
from snapshield.npu.model_runner import ModelRunner
from snapshield.npu.runtime import (
    ACCELERATED_PROVIDERS,
    RuntimeMode,
    get_available_providers,
)

DEFAULT_BENCHMARK_PROMPT = (
    "CONFIDENTIAL PATIENT ADMISSION:\n"
    "Patient: John Doe, Age: 45, MRN: 99482-B\n"
    "Address: 742 Evergreen Terrace, Springfield\n"
    "Admitted to City General Hospital for acute coronary syndrome.\n"
    "Contact: john.doe@hospital-synthetic.org, Emergency: +91 9876543210"
)


@dataclass
class BenchmarkSuiteResult:
    """Comparison results between CPU and verified accelerated provider."""
    cpu_result: BenchmarkResult
    accelerated_result: Optional[BenchmarkResult]
    is_accelerated_available: bool
    accelerated_provider_name: Optional[str]
    status_message: str
    workload_name: str
    iterations: int

    def to_dict(self) -> Dict[str, Any]:
        """Serializes benchmark comparison report."""
        return {
            "workload_name": self.workload_name,
            "iterations": self.iterations,
            "is_accelerated_available": self.is_accelerated_available,
            "status_message": self.status_message,
            "cpu": self.cpu_result.model_dump(),
            "accelerated": self.accelerated_result.model_dump() if self.accelerated_result else None,
        }


def _prepare_default_input_feed(model_path: Path) -> Dict[str, np.ndarray]:
    """Constructs a deterministic input feed for the benchmark workload."""
    tok_file = model_path.parent / "tokenizer.json"
    if tok_file.exists():
        try:
            import tokenizers
            tok = tokenizers.Tokenizer.from_file(str(tok_file))
            enc = tok.encode(DEFAULT_BENCHMARK_PROMPT)
            return {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
                "token_type_ids": np.zeros((1, len(enc.ids)), dtype=np.int64),
            }
        except Exception as exc:
            logger.debug("Failed to use fast tokenizer for benchmark feed: %s", exc)

    # Generic deterministic 64-token sequence
    seq_len = 64
    return {
        "input_ids": np.ones((1, seq_len), dtype=np.int64) * 101,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "token_type_ids": np.zeros((1, seq_len), dtype=np.int64),
    }


def benchmark_provider(
    model_path: Union[str, Path],
    input_feed: Optional[Dict[str, Any]] = None,
    mode: Union[str, RuntimeMode] = RuntimeMode.CPU,
    iterations: int = 10,
    warmup_iterations: int = 2,
) -> BenchmarkResult:
    """Executes granular benchmark measurements for a specific provider mode.

    Measures:
    - first inference (cold start)
    - warm inference (first post-warmup)
    - mean, median, min, max, std_dev
    - throughput (items/sec)
    """
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found for benchmark: {resolved_path}")

    if iterations < 1:
        raise ValueError(f"Benchmark iterations must be at least 1, got {iterations}")

    mode_enum = mode if isinstance(mode, RuntimeMode) else RuntimeMode(mode.upper())

    # Check accelerator availability upfront
    if mode_enum == RuntimeMode.ACCELERATED:
        available = get_available_providers()
        has_accel = any(
            p in available for p in ACCELERATED_PROVIDERS if p != "CPUExecutionProvider"
        )
        if not has_accel:
            return BenchmarkResult(
                execution_provider="Not available",
                sample_name=resolved_path.name,
                iterations=0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
                throughput_items_per_sec=0.0,
                first_inference_ms=None,
                warm_inference_ms=None,
                median_latency_ms=None,
                min_latency_ms=None,
                max_latency_ms=None,
                std_dev_ms=None,
                is_accelerated=False,
                status_message="Accelerated execution: Not available",
            )

    # Initialize model runner
    runner = ModelRunner(model_path=resolved_path, mode=mode_enum)
    runner.initialize()

    # If accelerated mode was requested but session fell back to CPU, report accurately
    if mode_enum == RuntimeMode.ACCELERATED and not runner.is_accelerated:
        runner.close()
        return BenchmarkResult(
            execution_provider="Not available",
            sample_name=resolved_path.name,
            iterations=0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            throughput_items_per_sec=0.0,
            first_inference_ms=None,
            warm_inference_ms=None,
            median_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            std_dev_ms=None,
            is_accelerated=False,
            status_message="Accelerated execution: Not available",
        )

    # Resolve input feed
    feed = input_feed or _prepare_default_input_feed(resolved_path)

    try:
        # 1. Cold start: First inference measurement
        t_cold_start = time.perf_counter()
        runner.run(feed)
        first_inference_ms = (time.perf_counter() - t_cold_start) * 1000.0

        # 2. Warmup iterations (burn-in caches, JIT, graph optimizers)
        for _ in range(max(0, warmup_iterations - 1)):
            runner.run(feed)

        # 3. Warm inference measurement (first post-warmup call)
        t_warm_start = time.perf_counter()
        runner.run(feed)
        warm_inference_ms = (time.perf_counter() - t_warm_start) * 1000.0

        # 4. Steady-state timed iterations
        steady_latencies_ms: List[float] = []
        for _ in range(iterations):
            t_iter = time.perf_counter()
            runner.run(feed)
            steady_latencies_ms.append((time.perf_counter() - t_iter) * 1000.0)

        # 5. Statistical aggregation (factual calculations)
        mean_lat = float(np.mean(steady_latencies_ms))
        median_lat = float(np.median(steady_latencies_ms))
        min_lat = float(np.min(steady_latencies_ms))
        max_lat = float(np.max(steady_latencies_ms))
        std_dev = float(np.std(steady_latencies_ms))
        p95_lat = float(np.percentile(steady_latencies_ms, 95))
        throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

        result = BenchmarkResult(
            execution_provider=runner.active_provider,
            sample_name=resolved_path.name,
            iterations=iterations,
            mean_latency_ms=round(mean_lat, 3),
            p95_latency_ms=round(p95_lat, 3),
            throughput_items_per_sec=round(throughput, 2),
            first_inference_ms=round(first_inference_ms, 3),
            warm_inference_ms=round(warm_inference_ms, 3),
            median_latency_ms=round(median_lat, 3),
            min_latency_ms=round(min_lat, 3),
            max_latency_ms=round(max_lat, 3),
            std_dev_ms=round(std_dev, 3),
            is_accelerated=runner.is_accelerated,
            status_message="Active",
            hardware_details={
                "npu_active": str(runner.is_npu_active),
                "accelerated": str(runner.is_accelerated),
                "fallback_occurred": str(runner.fallback_occurred),
            },
        )
        return result

    finally:
        runner.close()


def run_benchmark(
    model_path: Optional[Union[str, Path]] = None,
    input_feed: Optional[Dict[str, Any]] = None,
    iterations: int = 10,
    warmup_iterations: int = 2,
) -> BenchmarkSuiteResult:
    """Runs identical model/input workloads on CPU and verified accelerated provider.

    Guarantees:
    - Identical model weights and input feeds.
    - If accelerated provider is not available, clearly states:
      'Accelerated execution: Not available'.
    - Zero fabricated comparison metrics.
    """
    target_model = Path(model_path) if model_path else get_default_model_path()
    if target_model is None or not target_model.exists():
        raise FileNotFoundError(
            f"No ONNX model available for benchmark at: {target_model}. "
            "Please ensure default model is installed in snapshield/models/bert_small_pii/."
        )

    # 1. Benchmark CPU
    cpu_result = benchmark_provider(
        model_path=target_model,
        input_feed=input_feed,
        mode=RuntimeMode.CPU,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )

    # 2. Benchmark Accelerated Provider (if available)
    available_providers = get_available_providers()
    accel_candidates = [
        p for p in ACCELERATED_PROVIDERS
        if p != "CPUExecutionProvider" and p in available_providers
    ]

    accelerated_result: Optional[BenchmarkResult] = None
    is_accel_available = len(accel_candidates) > 0
    active_accel_name: Optional[str] = None

    if is_accel_available:
        try:
            candidate_res = benchmark_provider(
                model_path=target_model,
                input_feed=input_feed,
                mode=RuntimeMode.ACCELERATED,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
            if candidate_res.is_accelerated:
                accelerated_result = candidate_res
                active_accel_name = candidate_res.execution_provider
                status_msg = f"Accelerated execution active ({active_accel_name})."
            else:
                is_accel_available = False
                status_msg = "Accelerated execution: Not available"
        except Exception as exc:
            logger.warning("Accelerated benchmark failed: %s", exc)
            is_accel_available = False
            status_msg = "Accelerated execution: Not available"
    else:
        status_msg = "Accelerated execution: Not available"

    return BenchmarkSuiteResult(
        cpu_result=cpu_result,
        accelerated_result=accelerated_result,
        is_accelerated_available=is_accel_available,
        accelerated_provider_name=active_accel_name,
        status_message=status_msg,
        workload_name=target_model.name,
        iterations=iterations,
    )
