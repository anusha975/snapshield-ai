"""Unit tests for the hardware/runtime abstraction and model runner.

Validates:
- Factual environment detection (OS, arch, processor, ORT version, providers).
- Mode selection: AUTO, ACCELERATED, CPU.
- Provider fallback to CPU when accelerated providers are unavailable.
- Strict truthfulness: NEVER claims 'NPU active' unless confirmed by runtime.
- ModelRunner session lifecycle, fallback guarantees, and benchmark telemetry.
"""

import platform
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from snapshield.core.exceptions import HardwareAccelerationError
from snapshield.core.models import BenchmarkResult, RuntimeStatus
from snapshield.npu.model_runner import ModelRunner
from snapshield.npu.runtime import (
    ACCELERATED_PROVIDERS,
    NPU_PROVIDERS,
    RuntimeMode,
    detect_environment,
    get_active_provider,
    get_available_providers,
    get_runtime_status,
    health_check,
)


def test_factual_environment_detection():
    """Verifies that detected environment matches real host platform without fabrication."""
    env = detect_environment()

    assert env["os_name"] == platform.system()
    assert env["machine_arch"] == platform.machine()
    assert isinstance(env["processor_name"], str)
    assert len(env["processor_name"]) > 0
    assert isinstance(env["available_providers"], list)

    # In our local environment, ONNX Runtime is installed
    assert env["onnxruntime_version"] != "unavailable"
    assert "CPUExecutionProvider" in env["available_providers"]


def test_get_available_providers_does_not_assume_qnn():
    """Verifies get_available_providers reflects real runtime, never assuming QNN."""
    providers = get_available_providers()
    assert isinstance(providers, list)
    assert "CPUExecutionProvider" in providers

    # Verify we do not fabricate QNN if not actually installed in ort
    import onnxruntime as ort
    real_providers = ort.get_available_providers()
    assert providers == real_providers


def test_get_active_provider_cpu_mode():
    """Verifies CPU mode strictly selects CPUExecutionProvider regardless of environment."""
    provider = get_active_provider("CPU")
    assert provider == "CPUExecutionProvider"

    provider_enum = get_active_provider(RuntimeMode.CPU)
    assert provider_enum == "CPUExecutionProvider"

    provider_lower = get_active_provider("cpu")
    assert provider_lower == "CPUExecutionProvider"


def test_get_active_provider_auto_mode_current_host():
    """On the current host (x86_64, no QNN/DML installed), AUTO selects CPUExecutionProvider."""
    provider = get_active_provider("AUTO")
    # Since neither QNN nor DML is in available providers on this machine:
    assert provider == "CPUExecutionProvider"


def test_get_active_provider_auto_mode_with_qnn_available():
    """When QNN is available in runtime, AUTO selects QNNExecutionProvider."""
    with patch("snapshield.npu.runtime.get_available_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]):
        provider = get_active_provider("AUTO")
        assert provider == "QNNExecutionProvider"


def test_get_active_provider_auto_mode_with_dml_available():
    """When DirectML is available (and not QNN), AUTO selects DmlExecutionProvider."""
    with patch("snapshield.npu.runtime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
        provider = get_active_provider("AUTO")
        assert provider == "DmlExecutionProvider"


def test_get_active_provider_accelerated_mode_fallback():
    """When ACCELERATED is requested but no accelerator is installed, falls back to CPU."""
    with patch("snapshield.npu.runtime.get_available_providers", return_value=["CPUExecutionProvider"]):
        provider = get_active_provider("ACCELERATED")
        assert provider == "CPUExecutionProvider"


def test_never_claim_npu_active_on_current_host():
    """Verifies that runtime status NEVER claims NPU active unless QNN is confirmed."""
    status = get_runtime_status(mode="AUTO")
    assert isinstance(status, RuntimeStatus)

    # In this test environment (Intel PC), NPU is NOT active
    assert status.is_npu_active is False
    assert "NPU active" not in status.device_backend
    assert status.active_execution_provider == "CPUExecutionProvider"


def test_runtime_status_truthful_when_qnn_is_active():
    """Verifies that NPU active IS claimed when the runtime confirms QNN provider."""
    with patch("snapshield.npu.runtime.get_available_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]):
        status = get_runtime_status(mode="AUTO")
        assert status.is_npu_active is True
        assert status.is_accelerated is True
        assert status.active_execution_provider == "QNNExecutionProvider"
        assert "Qualcomm Snapdragon NPU active" in status.device_backend


def test_runtime_status_with_dml_gpu():
    """Verifies that DirectML GPU is reported accurately without claiming NPU."""
    with patch("snapshield.npu.runtime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
        status = get_runtime_status(mode="AUTO")
        assert status.is_npu_active is False
        assert status.is_accelerated is True
        assert status.active_execution_provider == "DmlExecutionProvider"
        assert "DirectML" in status.device_backend


def test_health_check_diagnostics():
    """Verifies health check produces complete factual diagnostics."""
    report = health_check()

    assert report["status"] == "healthy"
    assert report["os"] != ""
    assert report["architecture"] != ""
    assert report["processor"] != ""
    assert report["onnxruntime_installed"] is True
    assert isinstance(report["available_providers"], list)
    assert report["qnn_available"] is False  # Factual on this machine
    assert report["is_npu_active"] is False  # Factual on this machine
    assert len(report["diagnostics"]) > 0

    # Diagnostic messages must include explicit notice that NPU is not active
    diag_text = " ".join(report["diagnostics"])
    assert "NPU is NOT active" in diag_text


def test_model_runner_file_not_found():
    """ModelRunner raises FileNotFoundError if model path does not exist."""
    runner = ModelRunner(model_path="non_existent_model_12345.onnx", mode="AUTO")
    with pytest.raises(FileNotFoundError):
        runner.initialize()


def test_model_runner_candidate_provider_resolution():
    """Verifies candidate providers resolution across different modes."""
    runner_cpu = ModelRunner(model_path="dummy.onnx", mode="CPU")
    candidates_cpu = runner_cpu._resolve_candidate_providers()
    assert candidates_cpu == ["CPUExecutionProvider"]

    with patch("snapshield.npu.model_runner.get_available_providers", return_value=["QNNExecutionProvider", "CPUExecutionProvider"]):
        runner_auto = ModelRunner(model_path="dummy.onnx", mode="AUTO")
        candidates_auto = runner_auto._resolve_candidate_providers()
        assert candidates_auto[0] == "QNNExecutionProvider"
        assert candidates_auto[-1] == "CPUExecutionProvider"


def test_model_runner_fallback_to_cpu_when_accelerated_init_fails(tmp_path):
    """Verifies that ModelRunner falls back safely to CPU if primary provider fails."""
    dummy_model_file = tmp_path / "test_model.onnx"
    dummy_model_file.write_text("stub_onnx_content")

    mock_session_cpu = MagicMock()
    mock_session_cpu.get_providers.return_value = ["CPUExecutionProvider"]

    # First call with candidate fails; second call with CPU succeeds
    with patch("snapshield.npu.model_runner.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]), \
         patch("onnxruntime.InferenceSession", side_effect=[Exception("DML driver failure"), mock_session_cpu]):

        runner = ModelRunner(model_path=dummy_model_file, mode="ACCELERATED")
        runner.initialize()

        assert runner.fallback_occurred is True
        assert runner.active_provider == "CPUExecutionProvider"
        assert runner.is_npu_active is False
        assert runner.is_accelerated is False


def test_model_runner_benchmark_and_metadata(tmp_path):
    """Verifies ModelRunner benchmark telemetry and metadata reporting."""
    dummy_model_file = tmp_path / "benchmark_model.onnx"
    dummy_model_file.write_text("stub_onnx_content")

    mock_input = MagicMock()
    mock_input.name = "input_ids"
    mock_input.type = "tensor(int64)"
    mock_input.shape = [1, 128]

    mock_output = MagicMock()
    mock_output.name = "logits"
    mock_output.type = "tensor(float)"
    mock_output.shape = [1, 128, 15]

    mock_session = MagicMock()
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [MagicMock()]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        runner = ModelRunner(model_path=dummy_model_file, mode="CPU")
        runner.initialize()

        meta = runner.get_metadata()
        assert meta["initialized"] is True
        assert meta["active_provider"] == "CPUExecutionProvider"
        assert meta["is_npu_active"] is False
        assert len(meta["inputs"]) == 1
        assert meta["inputs"][0]["name"] == "input_ids"

        bench = runner.benchmark(input_feed={"input_ids": [1, 2, 3]}, iterations=5, warmup=1)
        assert isinstance(bench, BenchmarkResult)
        assert bench.iterations == 5
        assert bench.execution_provider == "CPUExecutionProvider"
        assert bench.mean_latency_ms >= 0.0

        # Context manager clean exit
        with runner:
            pass
        assert runner.session is None
