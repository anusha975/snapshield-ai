"""SnapShield AI NPU & Hardware Acceleration Module."""

from .benchmark import (
    BenchmarkSuiteResult,
    benchmark_provider,
    run_benchmark,
)
from .device import HardwareCapabilityReport, probe_hardware_capabilities
from .model_runner import ModelRunner
from .runtime import (
    RuntimeMode,
    detect_environment,
    get_active_provider,
    get_available_providers,
    get_runtime_status,
    health_check,
)
from .session import SafeInferenceSession

__all__ = [
    "BenchmarkSuiteResult",
    "HardwareCapabilityReport",
    "ModelRunner",
    "RuntimeMode",
    "SafeInferenceSession",
    "benchmark_provider",
    "detect_environment",
    "get_active_provider",
    "get_available_providers",
    "get_runtime_status",
    "health_check",
    "probe_hardware_capabilities",
    "run_benchmark",
]
