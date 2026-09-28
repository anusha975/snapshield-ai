"""Comprehensive tests for the compact Local AI Runtime panel.

Validates:
- Factual model detection and reporting.
- Truthful execution provider identification.
- Status determination: Verified / CPU fallback / unavailable.
- Actual inference latency reporting from scan telemetry.
- Actual benchmark reporting for CPU and Accelerated (or 'Accelerated execution: Not available').
- Strict avoidance of hardcoded numbers or unauthorized 'NPU' labels.
- Truthful handling of missing models or unavailable runtime.
- Streamlit UI component rendering stability.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.models import (
    BenchmarkResult,
    RiskAssessment,
    RiskLevel,
    RuntimeStatus,
    ScanResult,
    ScanTiming,
)
from snapshield.npu.benchmark import BenchmarkSuiteResult
from snapshield.npu.device import HardwareCapabilityReport
from snapshield.ui.components import (
    get_compact_runtime_info,
    render_compact_runtime_panel,
)


def test_get_compact_runtime_info_baseline_host():
    """Verifies compact runtime info on actual host environment."""
    info = get_compact_runtime_info()

    assert "model" in info
    assert "execution" in info
    assert "status" in info
    assert "inference" in info
    assert "cpu_benchmark" in info
    assert "accelerated_benchmark" in info

    # On this local environment with installed model:
    assert info["model"] == "bert-small-pii-detection-int8"
    assert info["execution"] == "CPUExecutionProvider"
    assert info["status"] == "CPU fallback"
    assert info["inference"] == "--"
    assert "Not available" in info["accelerated_benchmark"]


def test_get_compact_runtime_info_with_scan_result():
    """Verifies that actual scan AI inference latency is reported in the panel info."""
    scan = ScanResult(
        original_text="Patient confidential data",
        sanitized_text="Patient [REDACTED: CONFIDENTIAL]",
        risk_assessment=RiskAssessment(score=10.0, level=RiskLevel.LOW),
        timings=ScanTiming(ai_inference_ms=14.85),
        runtime_status=RuntimeStatus(
            os_name="Windows",
            machine_arch="AMD64",
            processor="Intel64",
            active_execution_provider="CPUExecutionProvider",
            is_accelerated=False,
            device_backend="CPUExecutionProvider",
        ),
    )

    info = get_compact_runtime_info(scan_result=scan)
    assert info["inference"] == "14.85 ms"


def test_get_compact_runtime_info_with_benchmark_suite():
    """Verifies that actual measured benchmark latencies are reported without hardcoding."""
    cpu_res = BenchmarkResult(
        execution_provider="CPUExecutionProvider",
        sample_name="model_quantized.onnx",
        iterations=5,
        mean_latency_ms=18.42,
        p95_latency_ms=21.0,
        throughput_items_per_sec=54.3,
    )
    suite = BenchmarkSuiteResult(
        cpu_result=cpu_res,
        accelerated_result=None,
        is_accelerated_available=False,
        accelerated_provider_name=None,
        status_message="Accelerated execution: Not available",
        workload_name="BERT-Small INT8",
        iterations=5,
    )

    info = get_compact_runtime_info(benchmark_suite=suite)
    assert info["cpu_benchmark"] == "18.42 ms"
    assert info["accelerated_benchmark"] == "Accelerated execution: Not available"


def test_get_compact_runtime_info_with_verified_accelerator():
    """Verifies status is 'Verified' when accelerated provider is active."""
    hw_report = HardwareCapabilityReport(
        os_name="Windows",
        os_release="11",
        machine_arch="ARM64",
        processor_name="Snapdragon X Elite",
        is_arm_architecture=True,
        is_qualcomm_snapdragon=True,
        available_onnx_providers=["QNNExecutionProvider", "CPUExecutionProvider"],
        qnn_provider_available=True,
        directml_provider_available=False,
        active_execution_provider="QNNExecutionProvider",
        is_accelerated=True,
        acceleration_summary="Qualcomm Snapdragon NPU Active",
    )

    cpu_res = BenchmarkResult(
        execution_provider="CPUExecutionProvider",
        sample_name="model_quantized.onnx",
        iterations=5,
        mean_latency_ms=19.5,
        p95_latency_ms=22.0,
        throughput_items_per_sec=51.0,
    )
    accel_res = BenchmarkResult(
        execution_provider="QNNExecutionProvider",
        sample_name="model_quantized.onnx",
        iterations=5,
        mean_latency_ms=4.25,
        p95_latency_ms=4.8,
        throughput_items_per_sec=235.3,
        is_accelerated=True,
    )
    suite = BenchmarkSuiteResult(
        cpu_result=cpu_res,
        accelerated_result=accel_res,
        is_accelerated_available=True,
        accelerated_provider_name="QNNExecutionProvider",
        status_message="Accelerated execution active (QNNExecutionProvider).",
        workload_name="BERT-Small INT8",
        iterations=5,
    )

    info = get_compact_runtime_info(
        hw_report=hw_report,
        benchmark_suite=suite,
    )

    assert info["status"] == "Verified"
    assert info["execution"] == "QNNExecutionProvider"
    assert info["cpu_benchmark"] == "19.50 ms"
    assert info["accelerated_benchmark"] == "4.25 ms"


def test_get_compact_runtime_info_when_model_missing():
    """Verifies status is 'unavailable' and model shows missing when model file is absent."""
    with patch("snapshield.models.is_default_model_installed", return_value=False), \
         patch("pathlib.Path.exists", return_value=False):
        info = get_compact_runtime_info()
        assert info["status"] == "unavailable"
        assert "Unavailable" in info["model"]


def test_get_compact_runtime_info_when_ort_unavailable():
    """Verifies status is 'unavailable' when ONNX Runtime is missing."""
    with patch("snapshield.npu.runtime.detect_environment", return_value={"onnxruntime_version": "unavailable", "available_providers": []}):
        info = get_compact_runtime_info()
        assert info["status"] == "unavailable"
        assert info["execution"] == "Unavailable"


def test_render_compact_runtime_panel_renders_safely():
    """Verifies render_compact_runtime_panel produces expected Streamlit UI elements."""
    with patch("streamlit.markdown") as mock_markdown:
        render_compact_runtime_panel()
        assert mock_markdown.called
        rendered_html = mock_markdown.call_args[0][0]
        assert "LOCAL AI RUNTIME" in rendered_html
        assert "Model:" in rendered_html
        assert "Execution:" in rendered_html
        assert "Status:" in rendered_html
        assert "Inference:" in rendered_html
        assert "Benchmark:" in rendered_html
        assert "CPU:" in rendered_html
        assert "Accelerated:" in rendered_html
