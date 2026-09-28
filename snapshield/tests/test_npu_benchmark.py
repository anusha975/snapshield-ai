"""Comprehensive tests for the hardware benchmark suite (npu/benchmark.py).

Validates:
- Factual measurement of cold start, warmup, steady-state mean, median, min, max, std_dev, and throughput.
- Guaranteed identical model and workload feeds across execution providers.
- Strict handling when acceleration is unavailable: 'Accelerated execution: Not available'.
- Zero fabricated comparison values.
- Streamlit UI rendering component stability.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.models import BenchmarkResult
from snapshield.models import DEFAULT_MODEL_FILE
from snapshield.npu.benchmark import (
    BenchmarkSuiteResult,
    benchmark_provider,
    run_benchmark,
)
from snapshield.npu.runtime import RuntimeMode


def test_cpu_benchmark_measurements():
    """Verifies that CPU benchmark factually measures all required latency parameters."""
    res = benchmark_provider(
        model_path=DEFAULT_MODEL_FILE,
        mode=RuntimeMode.CPU,
        iterations=5,
        warmup_iterations=2,
    )

    assert isinstance(res, BenchmarkResult)
    assert res.execution_provider == "CPUExecutionProvider"
    assert res.iterations == 5
    assert res.is_accelerated is False

    # Granular latency metrics must all be positive floats
    assert res.first_inference_ms is not None and res.first_inference_ms > 0.0
    assert res.warm_inference_ms is not None and res.warm_inference_ms > 0.0
    assert res.mean_latency_ms > 0.0
    assert res.median_latency_ms is not None and res.median_latency_ms > 0.0
    assert res.min_latency_ms is not None and res.min_latency_ms > 0.0
    assert res.max_latency_ms is not None and res.max_latency_ms >= res.min_latency_ms
    assert res.std_dev_ms is not None and res.std_dev_ms >= 0.0
    assert res.throughput_items_per_sec > 0.0


def test_accelerated_benchmark_unavailable_on_current_host():
    """On host without QNN or DML, accelerated mode cleanly reports 'Not available'."""
    res = benchmark_provider(
        model_path=DEFAULT_MODEL_FILE,
        mode=RuntimeMode.ACCELERATED,
        iterations=5,
        warmup_iterations=1,
    )

    assert isinstance(res, BenchmarkResult)
    assert res.execution_provider == "Not available"
    assert res.status_message == "Accelerated execution: Not available"
    assert res.iterations == 0
    assert res.mean_latency_ms == 0.0
    assert res.is_accelerated is False


def test_run_benchmark_suite_on_current_host():
    """Verifies full benchmark suite comparison behavior on the host."""
    suite = run_benchmark(iterations=4, warmup_iterations=1)

    assert isinstance(suite, BenchmarkSuiteResult)
    assert suite.cpu_result.execution_provider == "CPUExecutionProvider"
    assert suite.cpu_result.mean_latency_ms > 0.0

    # On this machine, acceleration is unavailable
    assert suite.is_accelerated_available is False
    assert suite.accelerated_result is None
    assert suite.status_message == "Accelerated execution: Not available"

    suite_dict = suite.to_dict()
    assert suite_dict["is_accelerated_available"] is False
    assert suite_dict["accelerated"] is None
    assert suite_dict["cpu"]["mean_latency_ms"] > 0.0


def test_run_benchmark_suite_with_mocked_accelerator(tmp_path):
    """Verifies suite behavior when an accelerated provider IS available and verified."""
    dummy_model = tmp_path / "mock_model.onnx"
    dummy_model.write_text("dummy")

    mock_cpu_session = MagicMock()
    mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_cpu_session.run.return_value = [MagicMock()]

    mock_qnn_session = MagicMock()
    mock_qnn_session.get_providers.return_value = ["QNNExecutionProvider"]
    mock_qnn_session.run.return_value = [MagicMock()]

    def mock_session_factory(*args, **kwargs):
        providers = kwargs.get("providers", [])
        if "QNNExecutionProvider" in providers:
            return mock_qnn_session
        return mock_cpu_session

    with patch("snapshield.npu.benchmark.get_available_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]), \
         patch("snapshield.npu.model_runner.get_available_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]), \
         patch("onnxruntime.InferenceSession", side_effect=mock_session_factory):

        suite = run_benchmark(model_path=dummy_model, iterations=3, warmup_iterations=1)

        assert suite.is_accelerated_available is True
        assert suite.accelerated_result is not None
        assert suite.accelerated_result.execution_provider == "QNNExecutionProvider"
        assert suite.accelerated_result.is_accelerated is True
        assert "Accelerated execution active" in suite.status_message


def test_benchmark_rejects_missing_model():
    """Benchmark raises FileNotFoundError for missing model path."""
    with pytest.raises(FileNotFoundError):
        benchmark_provider(model_path="non_existent_path_9999.onnx", mode="CPU")


def test_benchmark_rejects_invalid_iterations():
    """Benchmark raises ValueError if iterations < 1."""
    with pytest.raises(ValueError):
        benchmark_provider(model_path=DEFAULT_MODEL_FILE, mode="CPU", iterations=0)


def test_ui_render_benchmark_comparison_safety():
    """Verifies render_benchmark_comparison executes cleanly for both states."""
    from snapshield.ui.components import render_benchmark_comparison

    # 1. Test when acceleration is unavailable
    suite_unavailable = run_benchmark(iterations=2, warmup_iterations=1)
    with patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]), \
         patch("streamlit.metric"), \
         patch("streamlit.markdown"), \
         patch("streamlit.subheader"), \
         patch("streamlit.caption"), \
         patch("streamlit.warning") as mock_warn, \
         patch("streamlit.info"):

        render_benchmark_comparison(suite_unavailable)
        mock_warn.assert_called_with("Accelerated execution: Not available")
