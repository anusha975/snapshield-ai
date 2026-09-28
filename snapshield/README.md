# SnapShield AI (🛡️)

> **On-Device Privacy & Security Checkpoint**  
> Analyzes documents, text, and screenshots before content is sent to external AI services.

---

## 1. Product Overview

SnapShield AI acts as an on-device egress perimeter firewall for AI workloads. Rather than allowing raw prompts, sensitive enterprise documents, or user data to leave the device unexamined, SnapShield AI intercepts and evaluates them locally.

### Primary Workflow
```
Input (Text / Document / Screenshot)
  │
  ▼
Extraction (pdfplumber / Pillow)
  │
  ▼
Local OCR (if image/screenshot)
  │
  ▼
Hybrid Detection (Deterministic Regex + Luhn Checksum + Local Contextual AI)
  │
  ▼
Risk Assessment (Composite score 0-100, Tier: Low/Med/High/Critical)
  │
  ▼
Policy Decision (STRICT / BALANCED / PERMISSIVE ➔ BLOCK / REDACT / ALLOW / REVIEW)
  │
  ├───[BLOCK / REVIEW] ➔ Egress Terminated / Flagged
  │
  └───[REDACT / ALLOW] ➔ Safe Sanitized Output (SHA-256 Verified)
                                │
                                ▼
                      Optional External AI (Egress)
```

---

## 2. Core Architectural Principles & Invariants

1. **Zero External Raw Transmission**: Never send original sensitive content to an external LLM. Only cryptographically verified sanitized outputs may pass.
2. **Zero Raw Logging**: Never log raw sensitive text or tokens into standard logs or audit trails.
3. **No Fabricated AI Confidence**: Report authentic model confidence metrics; never synthesize arbitrary high confidence scores.
4. **Truthful Hardware Detection**: Never fabricate Snapdragon or NPU performance. Probe available ONNX execution providers (`QNNExecutionProvider`, `DmlExecutionProvider`) and explicitly fallback to `CPUExecutionProvider` when acceleration is not available.
5. **Decoupled Boundary**: Maintain absolute separation between the original raw input and sanitized output artifacts.
6. **Robust Error Handling**: Handle failures gracefully without silently swallowing exceptions.

---

## 3. Project Structure

```
snapshield/
├── app.py                     # Streamlit application entrypoint
├── requirements.txt           # Project dependencies (Streamlit, pdfplumber, Pillow, ONNX, etc.)
├── README.md                  # System architecture, guidelines, and setup
├── .env.example               # Template environment configuration
├── .gitignore                 # Version control exclusions
│
├── core/                      # Domain models, exceptions, and privacy-safe logging
│   ├── types.py               # Strongly-typed schemas (Pydantic)
│   ├── exceptions.py          # Structured domain exception hierarchy
│   └── logger.py              # Scrubbed, privacy-preserving logger
│
├── services/                  # Business logic services
│   ├── extractor.py           # Multi-format document extraction (Text, PDF, Image)
│   ├── ocr.py                 # Local OCR interface with truthful confidence handling
│   ├── detector.py            # Hybrid detection (Regex + Luhn checksum + Contextual)
│   ├── risk.py                # Multi-category weighted risk scoring
│   ├── policy.py              # Configurable privacy profile enforcement
│   ├── redactor.py            # Masking engine preserving original/sanitized separation
│   └── llm_gateway.py         # Downstream egress gatekeeper
│
├── npu/                       # Hardware acceleration detection and session management
│   ├── device.py              # Hardware probing (Qualcomm Snapdragon, DirectML, CPU fallback)
│   └── session.py             # ONNX Runtime session manager with CPU fallback
│
├── ui/                        # Presentation layer (Streamlit)
│   ├── layout.py              # Page layout, custom CSS, header, and sidebar
│   ├── components.py          # Reusable KPI scorecards, risk badges, findings table
│   └── views.py               # Input ingestion and checkpoint tabs
│
├── config/                    # Configuration and policy definitions
│   ├── settings.py            # Type-safe pydantic settings
│   └── policies.yaml          # Risk weights and profile thresholds
│
├── models/                    # Local ONNX model specifications
│   └── README.md              # ONNX quantization and export instructions
│
├── data/                      # Synthetic evaluation datasets
│   └── synthetic_samples.json # Synthetic credit cards, SSNs, and developer configs
│
├── tests/                     # Unit and integration test suite
│   ├── test_imports.py        # Import and submodule integrity
│   ├── test_npu_detection.py  # Hardware probing & CPU fallback tests
│   └── test_synthetic_data.py # Synthetic sample validation and end-to-end pipeline test
│
└── assets/                    # Media, icons, and diagrams
    └── README.md
```

---

## 4. Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Clone or navigate to the repository
cd "snapshield"

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/ -v
```

### Running the Application
```bash
streamlit run app.py
```

---

## 5. Hardware Acceleration (Snapdragon & DirectML)

SnapShield AI probes execution providers via `onnxruntime.get_available_providers()`:
- If a Qualcomm Snapdragon platform is detected and `QNNExecutionProvider` is available, hardware NPU acceleration is verified.
- If running on a Windows system with DirectML support, `DmlExecutionProvider` is utilized.
- In all other environments, `CPUExecutionProvider` is automatically engaged as a verified, stable fallback.
