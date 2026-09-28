"""Verification tests for SnapShield AI package imports."""

import sys
from pathlib import Path
import pytest

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_import_core():
    """Verify that snapshield.core imports cleanly."""
    import snapshield.core
    from snapshield.core.types import CheckpointResult, Finding, PolicyAction
    from snapshield.core.exceptions import SnapShieldError
    from snapshield.core.logger import logger

    assert CheckpointResult is not None
    assert Finding is not None
    assert PolicyAction.BLOCK == "BLOCK"
    assert logger is not None


def test_import_config():
    """Verify that snapshield.config imports cleanly."""
    from snapshield.config.settings import get_settings
    from snapshield.config.policies import get_policy_manager, PolicyConfigManager

    settings = get_settings()
    assert settings.policy_profile in ["Personal", "Developer", "STRICT", "BALANCED", "PERMISSIVE"]
    assert settings.default_policy_action in ["BLOCK", "REDACT", "ALLOW", "REVIEW"]

    pm = get_policy_manager()
    assert "Personal" in pm.list_profiles()
    assert "Developer" in pm.list_profiles()


def test_import_npu():
    """Verify that snapshield.npu imports and probes hardware without errors."""
    from snapshield.npu.device import probe_hardware_capabilities

    hw = probe_hardware_capabilities()
    assert hw.active_execution_provider is not None
    assert isinstance(hw.available_onnx_providers, list)
    # Ensure CPU fallback exists
    assert "CPUExecutionProvider" in hw.available_onnx_providers or hw.active_execution_provider == "CPUExecutionProvider"


def test_import_services():
    """Verify that snapshield.services imports cleanly."""
    from snapshield.services.detector import HybridDetectionEngine, luhn_checksum_validate
    from snapshield.services.extractor import DocumentExtractor
    from snapshield.services.ocr import LocalOCRService
    from snapshield.services.policy import PolicyEngine
    from snapshield.services.redactor import RedactionEngine
    from snapshield.services.risk import RiskAssessmentEngine
    from snapshield.services.llm_gateway import ExternalLLMGateway

    assert HybridDetectionEngine is not None
    assert DocumentExtractor is not None
    assert LocalOCRService is not None
    assert PolicyEngine is not None
    assert RedactionEngine is not None
    assert RiskAssessmentEngine is not None
    assert ExternalLLMGateway is not None


def test_import_ui():
    """Verify that snapshield.ui imports cleanly."""
    from snapshield.ui.components import render_findings_table
    from snapshield.ui.layout import setup_page_config
    from snapshield.ui.views import render_checkpoint_results

    assert render_findings_table is not None
    assert setup_page_config is not None
    assert render_checkpoint_results is not None
