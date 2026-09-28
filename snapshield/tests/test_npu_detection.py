"""Tests for hardware probing and Snapdragon/NPU detection integrity.

Validates:
- Rule 4: Never fabricate Snapdragon/NPU performance.
- Rule 5: Never claim accelerated execution unless runtime verification confirms it.
- Rule 6: CPU fallback must always exist.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snapshield.npu.device import probe_hardware_capabilities


def test_npu_detection_truthfulness():
    """Verifies that hardware capabilities report reflects real environment without fabrication."""
    report = probe_hardware_capabilities()

    assert report.os_name != ""
    assert report.machine_arch != ""
    assert report.active_execution_provider != ""

    # If active provider claims acceleration, it must be supported in available ONNX providers
    if report.is_accelerated:
        assert report.active_execution_provider in report.available_onnx_providers
        assert report.active_execution_provider in ["QNNExecutionProvider", "DmlExecutionProvider"]
    else:
        assert report.active_execution_provider == "CPUExecutionProvider"
        assert "CPU fallback" in report.acceleration_summary


def test_cpu_fallback_guarantee():
    """Verifies that CPU execution fallback is always accounted for."""
    report = probe_hardware_capabilities()
    assert "CPUExecutionProvider" in report.available_onnx_providers or report.active_execution_provider == "CPUExecutionProvider"
