# Local AI Models for SnapShield AI

This directory contains on-device, privacy-preserving AI models used by SnapShield AI for contextual entity and sensitive data detection. All models operate entirely locally with zero external network calls.

---

## Integrated Model Specification

### 1. Model Name
`bert-small-pii-detection-int8` (Base: `bert-small-pii-detection`)

### 2. Source
* **Base Model:** [`gravitee-io/bert-small-pii-detection`](https://huggingface.co/gravitee-io/bert-small-pii-detection)
* **ONNX Optimized & Quantized Distribution:** [`onnx-community/bert-small-pii-detection-ONNX`](https://huggingface.co/onnx-community/bert-small-pii-detection-ONNX)
* **Training Datasets:** `beki/privy`, `gretelai/synthetic_pii_finance_multilingual`, `eriktks/conll2003`

### 3. License
**Apache License 2.0**  
Permissive open-source license permitting commercial, non-commercial, modification, and on-device distribution without proprietary restrictions.

### 4. Version
* **Model Version:** `v1.0.0`
* **ONNX Opset:** `17`
* **Quantization Scheme:** INT8 static/dynamic quantization (8-bit weights and activations)

### 5. Input Format
The model accepts three standard BERT-style integer tensor inputs:
* **`input_ids`**: `int64[batch_size, sequence_length]`
  * WordPiece token identifiers indexed against the 30,522-entry vocabulary.
* **`attention_mask`**: `int64[batch_size, sequence_length]`
  * Binary mask where `1` indicates real content tokens and `0` indicates padding tokens.
* **`token_type_ids`**: `int64[batch_size, sequence_length]`
  * Segment indicators (`0` for single-sequence classification).

### 6. Output Format
* **`logits`**: `float32[batch_size, sequence_length, 49]`
  * Raw, unnormalized prediction scores for 49 token-classification classes representing 24 PII/NER entity types in BIO (Begin, Inside, Outside) tagging notation.
  * Supported target classes include:
    * `PERSON` (`B-PERSON`, `I-PERSON`)
    * `ORGANIZATION` (`B-ORGANIZATION`, `I-ORGANIZATION`)
    * `LOCATION` / `COORDINATE` (`B-LOCATION`, `I-LOCATION`)
    * `PASSWORD` / `FINANCIAL` / `CREDIT_CARD` / `IBAN_CODE` / `US_BANK_NUMBER`
    * `PHONE_NUMBER` / `EMAIL_ADDRESS` / `IP_ADDRESS` / `MAC_ADDRESS`
    * `US_SSN` / `US_PASSPORT` / `US_DRIVER_LICENSE` / `US_ITIN`

### 7. Preprocessing
1. **Tokenization:**
   * Text is tokenized into WordPiece subword tokens via fast Rust-backed `tokenizers.Tokenizer` (with regex fallback).
   * Exact character span offsets `(start_char, end_char)` in the original text are preserved for every token.
2. **Context Window Alignment:**
   * Sequences exceeding the model context window (default 512 tokens) are truncated to prevent tensor overflow.
3. **Tensor Construction:**
   * Subword tokens are converted into 64-bit integer NumPy arrays with an added batch dimension `[1, seq_len]`.

### 8. Postprocessing
1. **Softmax Normalization:**
   * Numerically stable Softmax is computed along the class dimension to derive true class probabilities:
     $$\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$
2. **Argmax Classification:**
   * Highest-probability label ID is determined per token.
3. **Category Mapping:**
   * Model classes are normalized to SnapShield standard privacy categories (`PERSON`, `ORGANIZATION`, `PRIVATE_LOCATION`, `CONFIDENTIAL_BUSINESS_INFORMATION`, `PATIENT_IDENTIFIER`).
4. **BIO Span Merging:**
   * Contiguous `B-` and `I-` tokens of the same category are consolidated into complete entity spans.
   * Span character boundaries `(start, end)` are mapped back to original document text coordinates.
   * Span confidence is computed as the arithmetic mean of contributing token probabilities.
5. **Domain Object Emission:**
   * Emits strongly-typed `Detection` objects containing masked display values, verified provider metadata, severity rating, and explanation.

### 9. Model Size & Footprint
* **Model Weights (`model_quantized.onnx`):** `27.42 MB` (28,748,527 bytes)
* **Tokenizer Configuration (`tokenizer.json`):** `712 KB`
* **Model Configuration (`config.json`):** `3.4 KB`
* **Total Storage Footprint:** `~28.2 MB`
* **Memory Footprint at Runtime:** `< 85 MB RAM`

### 10. Runtime Requirements
* **Python Runtime:** Python >= 3.9 (tested on Python 3.13)
* **Inference Engine:** `onnxruntime` >= 1.15.0
* **Numerical Processing:** `numpy` >= 1.20.0
* **Tokenizer (Optional / Recommended):** `tokenizers` >= 0.13.0
* **Supported Execution Providers:**
  * `CPUExecutionProvider`: Universal CPU fallback (verified on x86_64, AMD64, ARM64).
  * `DmlExecutionProvider`: DirectML acceleration on Windows (Intel/AMD/NVIDIA GPUs).
  * `QNNExecutionProvider`: Qualcomm Snapdragon NPU (HTP backend on ARM64 Snapdragon platforms).
* **Truthfulness Invariant:**
  * Execution provider acceleration is orchestrated through `ModelRunner`.
  * Hardware acceleration is NEVER claimed unless confirmed by the runtime session (`session.get_providers()`).

---

## Directory Structure
```
snapshield/models/
├── README.md                          # This specification document
├── __init__.py                        # Model registry and path helpers
└── bert_small_pii/
    ├── config.json                    # Model architecture and label mapping
    ├── model_quantized.onnx           # INT8 quantized ONNX model graph
    └── tokenizer.json                 # Fast WordPiece tokenizer configuration
```
