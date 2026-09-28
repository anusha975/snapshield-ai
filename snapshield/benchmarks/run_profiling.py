"""Comprehensive profiling and benchmark suite for SnapShield AI.

Measures:
1. Model Session Loading (Cold vs Cached)
2. Hardware Capabilities Probing (Cold vs Cached)
3. Deterministic Structured Detection Latency
4. Credential & Secret Detection Latency
5. Local AI Inference Latency (Cold vs Warm)
6. OCR Extraction Latency (Cold vs Ephemeral Cached)
7. End-to-End Privacy Checkpoint Scan Latency (Cold vs Warm)
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from PIL import Image

from snapshield.core.ai_detector import LocalAIDetector
from snapshield.core.code_detectors import detect_credentials
from snapshield.core.detectors import detect_structured
from snapshield.core.logger import logger
from snapshield.core.models import InputDocument
from snapshield.core.ocr import LocalOCRProvider, OCROutput, clear_ocr_cache
from snapshield.models import get_default_model_path
from snapshield.npu.device import clear_hardware_cache, probe_hardware_capabilities
from snapshield.npu.model_runner import clear_model_session_cache, get_model_session_cache_size
from snapshield.services.scan_service import scan_document, scan_image

# Suppress logger during benchmarking output
logger.setLevel(logging.WARNING)



def profile_model_loading() -> dict[str, float]:
    """Measures model session initialization time before and after caching."""
    model_path = get_default_model_path()
    if not model_path or not model_path.exists():
        return {"cold_load_ms": 0.0, "cached_load_ms": 0.0, "speedup": 1.0}

    # 1. Measure Cold Load (Clearing Session Cache)
    clear_model_session_cache()
    t0 = time.perf_counter()
    det1 = LocalAIDetector(model_path=model_path)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Measure Cached Load
    t0 = time.perf_counter()
    det2 = LocalAIDetector(model_path=model_path)
    cached_ms = (time.perf_counter() - t0) * 1000.0

    speedup = cold_ms / max(cached_ms, 0.001)
    return {
        "cold_load_ms": round(cold_ms, 3),
        "cached_load_ms": round(cached_ms, 3),
        "speedup": round(speedup, 1),
    }


def profile_hardware_probing() -> dict[str, float]:
    """Measures hardware detection latency before and after caching."""
    clear_hardware_cache()
    t0 = time.perf_counter()
    _ = probe_hardware_capabilities()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _ = probe_hardware_capabilities()
    cached_ms = (time.perf_counter() - t0) * 1000.0

    speedup = cold_ms / max(cached_ms, 0.0001)
    return {
        "cold_probe_ms": round(cold_ms, 4),
        "cached_probe_ms": round(cached_ms, 4),
        "speedup": round(speedup, 1),
    }


def profile_detection_engines() -> dict[str, float]:
    """Measures throughput and latency of deterministic and credential detectors."""
    sample_text = (
        "CONFIDENTIAL EMPLOYEE FILE\n"
        "Employee Name: Rajesh Kumar\n"
        "Aadhaar Number: 3675 9834 1218\n"
        "Permanent Account Number (PAN): ABCDE1234F\n"
        "Contact: +91 98765 43210, alternate: +91 9123456789\n"
        "Email: rajesh.kumar@enterprise-solutions.internal\n"
        "Salary Account: 123456789012 at State Bank of India\n"
        "IFSC Code: SBIN0001234\n"
        "Production Secret Key: sk-live-abc1234567890def1234567890gh\n"
        "Database: postgresql://admin:P@ssw0rd123!@db.internal:5432/payroll\n"
        "AWS Access: AKIAIOSFODNN7EXAMPLE\n"
    )

    # 1. Structured Detection
    t0 = time.perf_counter()
    for _ in range(50):
        _ = detect_structured(sample_text)
    struct_avg_ms = ((time.perf_counter() - t0) / 50) * 1000.0

    # 2. Credential Detection
    t0 = time.perf_counter()
    for _ in range(50):
        _ = detect_credentials(sample_text)
    cred_avg_ms = ((time.perf_counter() - t0) / 50) * 1000.0

    return {
        "structured_detection_ms": round(struct_avg_ms, 3),
        "credential_detection_ms": round(cred_avg_ms, 3),
    }


def profile_ocr_caching() -> dict[str, float]:
    """Measures OCR execution latency before and after ephemeral caching."""
    # Create synthetic test image
    img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # Custom fallback for deterministic benchmarking without external tesseract dependency
    def mock_extractor(im: Image.Image, p_idx: int) -> OCROutput:
        time.sleep(0.05)  # Simulate 50ms OCR processing
        return OCROutput(text="SAMPLE OCR TEXT", provider_name="mock_ocr", processing_time_ms=50.0)

    provider = LocalOCRProvider(custom_fallback=mock_extractor)
    clear_ocr_cache()

    # Cold OCR run
    t0 = time.perf_counter()
    _ = provider.extract_from_image(raw_bytes)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Cached OCR run
    t0 = time.perf_counter()
    _ = provider.extract_from_image(raw_bytes)
    cached_ms = (time.perf_counter() - t0) * 1000.0

    speedup = cold_ms / max(cached_ms, 0.001)
    return {
        "cold_ocr_ms": round(cold_ms, 2),
        "cached_ocr_ms": round(cached_ms, 4),
        "speedup": round(speedup, 1),
    }


def profile_end_to_end_pipeline() -> dict[str, float]:
    """Measures full scan pipeline latency from raw input to sanitized output."""
    sample_text = (
        "CONFIDENTIAL EMPLOYEE FILE\n"
        "Employee Name: Rajesh Kumar\n"
        "Aadhaar Number: 3675 9834 1218\n"
        "Permanent Account Number (PAN): ABCDE1234F\n"
        "Contact: +91 98765 43210\n"
        "Email: rajesh.kumar@enterprise-solutions.internal\n"
        "Salary Account: 123456789012\n"
        "IFSC Code: SBIN0001234\n"
        "API Key: sk-live-abc1234567890def1234567890gh\n"
    )

    # First run (Cold)
    res_cold = scan_document(sample_text, policy_mode="Personal", ai_enabled=True)

    # Repeat run (Warm)
    res_warm = scan_document(sample_text, policy_mode="Personal", ai_enabled=True)

    return {
        "cold_total_ms": round(res_cold.timings.total_ms, 2),
        "warm_total_ms": round(res_warm.timings.total_ms, 2),
        "cold_ai_ms": round(res_cold.timings.ai_inference_ms, 2),
        "warm_ai_ms": round(res_warm.timings.ai_inference_ms, 2),
        "structured_ms": round(res_warm.timings.structured_detection_ms, 2),
        "risk_ms": round(res_warm.timings.risk_ms, 2),
        "policy_ms": round(res_warm.timings.policy_ms, 2),
        "redaction_ms": round(res_warm.timings.redaction_ms, 2),
    }


if __name__ == "__main__":
    print("\n========================================================")
    print("      SnapShield AI Profiling & Telemetry Benchmark      ")
    print("========================================================\n")
    
    model_res = profile_model_loading()
    print("1. ONNX Model Session Initialization:")
    print(f"   - Cold Session Creation (Protobuf load): {model_res['cold_load_ms']} ms")
    print(f"   - Cached Session Lookup (Memory pool):   {model_res['cached_load_ms']} ms")
    print(f"   - Optimization Speedup:                  {model_res['speedup']}x\n")

    hw_res = profile_hardware_probing()
    print("2. Hardware Capability Probing:")
    print(f"   - Cold Hardware Probe (OS & Drivers):    {hw_res['cold_probe_ms']} ms")
    print(f"   - Cached Probe (LRU Cache):              {hw_res['cached_probe_ms']} ms")
    print(f"   - Optimization Speedup:                  {hw_res['speedup']}x\n")

    det_res = profile_detection_engines()
    print("3. Deterministic Detectors Throughput:")
    print(f"   - Structured (Verhoeff/PAN/Phone/Email): {det_res['structured_detection_ms']} ms/doc")
    print(f"   - Credentials & Secrets (Regex module):  {det_res['credential_detection_ms']} ms/doc\n")

    ocr_res = profile_ocr_caching()
    print("4. Ephemeral Bounded OCR Caching:")
    print(f"   - Cold OCR Extraction:                   {ocr_res['cold_ocr_ms']} ms")
    print(f"   - Ephemeral Cache Hit:                   {ocr_res['cached_ocr_ms']} ms")
    print(f"   - Optimization Speedup:                  {ocr_res['speedup']}x\n")

    e2e_res = profile_end_to_end_pipeline()
    print("5. End-to-End Privacy Checkpoint Scan:")
    print(f"   - Cold Document Scan Total:              {e2e_res['cold_total_ms']} ms")
    print(f"   - Warm Document Scan Total:              {e2e_res['warm_total_ms']} ms")
    print(f"   - AI Contextual Inference (Local ONNX):  {e2e_res['warm_ai_ms']} ms")
    print(f"   - Structured & Credential Detection:     {e2e_res['structured_ms']} ms")
    print(f"   - Risk Assessment Engine:                {e2e_res['risk_ms']} ms")
    print(f"   - Policy Enforcement:                    {e2e_res['policy_ms']} ms")
    print(f"   - Cryptographic Sanitization & Redactor: {e2e_res['redaction_ms']} ms")
    print("\n========================================================\n")
