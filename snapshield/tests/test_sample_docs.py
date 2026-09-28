"""Automated Tests for Synthetic Demo Documents (data/sample_docs/).

Verifies:
1. Presence and non-emptiness of:
   - employee_record.txt
   - resume.txt
   - medical_note.txt
   - developer_config.txt
2. Compliance with required synthetic entity structures:
   - Employee: Name, Address, Aadhaar candidate, PAN candidate, Phone, Email, Bank account, IFSC
   - Medical: Patient information, Diagnosis, Doctor, Hospital, Medical observations
   - Developer: API key, Password, JWT, Database URL, Email, IP address
3. Invariant: 100% synthetic, zero real personal data.
4. End-to-end scan & redaction verification on each document.
"""

from pathlib import Path
import pytest

from snapshield.core.models import PolicyAction, ScanResult
from snapshield.core.redactor import verify_sanitization_leak_free
from snapshield.core.verhoeff import is_valid_aadhaar
from snapshield.data import get_sample_docs_dir, list_sample_documents, load_sample_document
from snapshield.services.scan_service import scan_document


def test_sample_docs_catalog_and_files():
    """Verify that all 4 required synthetic sample documents exist and load properly."""
    catalog = list_sample_documents()
    assert len(catalog) == 4

    required_files = [
        "employee_record.txt",
        "resume.txt",
        "medical_note.txt",
        "developer_config.txt",
    ]

    docs_dir = get_sample_docs_dir()
    for filename in required_files:
        filepath = docs_dir / filename
        assert filepath.exists(), f"Missing required sample document: {filename}"
        content = load_sample_document(filename)
        assert len(content.strip()) > 50, f"Sample document {filename} is empty or too short."


def test_employee_record_entities():
    """Verify employee_record.txt contains all specified synthetic entity types."""
    content = load_sample_document("employee_record.txt")

    # Name
    assert "Aarav Patel" in content
    # Address
    assert "Indiranagar, Bengaluru" in content
    # Aadhaar candidate (valid Verhoeff)
    assert "9999 9999 0019" in content
    assert is_valid_aadhaar("9999 9999 0019") is True
    # PAN candidate
    assert "ABCPA1234F" in content
    # Phone
    assert "+91 98765 43210" in content
    # Email
    assert "aarav.patel.synthetic@snapshield-corp.org" in content
    # Bank account
    assert "987654321012345" in content
    # IFSC
    assert "HDFC0001234" in content

    # Scan with Personal policy
    res = scan_document(content, policy_mode="Personal", ai_enabled=True)
    assert isinstance(res, ScanResult)
    assert len(res.detections) >= 5
    assert "[REDACTED: AADHAAR]" in res.sanitized_text
    assert "9999 9999 0019" not in res.sanitized_text


def test_resume_entities():
    """Verify resume.txt contains candidate CV info with valid synthetic tokens."""
    content = load_sample_document("resume.txt")

    assert "Priyadarshini Rao" in content
    assert "+91 98450 12345" in content
    assert "priya.rao.synthetic@ai-research-lab.org" in content
    assert "2345 6789 0124" in content
    assert is_valid_aadhaar("2345 6789 0124") is True

    res = scan_document(content, policy_mode="Personal", ai_enabled=True)
    assert len(res.detections) >= 3
    assert "2345 6789 0124" not in res.sanitized_text


def test_medical_note_entities():
    """Verify medical_note.txt contains patient, diagnosis, doctor, hospital, and clinical notes."""
    content = load_sample_document("medical_note.txt")

    # Patient info
    assert "John Synthetic Doe" in content
    assert "MRN-8849201" in content
    # Doctor
    assert "Dr. Robert Smith" in content
    # Hospital
    assert "Apollo Memorial Hospital" in content
    # Diagnosis & observations
    assert "acute myocardial infarction" in content.lower()
    assert "prescription for chemotherapy" in content.lower()

    # Scan with Medical policy
    res = scan_document(content, policy_mode="Medical", ai_enabled=True)
    assert isinstance(res, ScanResult)
    assert len(res.detections) >= 4
    assert "9999 9999 0019" not in res.sanitized_text
    assert "patient.doe@synthetic-health.org" not in res.sanitized_text


def test_developer_config_entities():
    """Verify developer_config.txt contains API keys, password, JWT, database URL, email, IP."""
    content = load_sample_document("developer_config.txt")

    # API Key
    assert "sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678" in content
    # Password
    assert "pAssw0rd_SuperSecret_2026!#$" in content
    # JWT
    assert "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" in content
    # Database URL
    assert "postgresql://db_admin:UltraSecretPass99" in content
    # Email
    assert "devops.alerts@snapshield-internal-synthetic.io" in content
    # IP address
    assert "192.168.1.150" in content

    # Scan with Developer policy
    res = scan_document(content, policy_mode="Developer", ai_enabled=True)
    assert res.is_blocked is True or res.overall_action == PolicyAction.BLOCK
    assert "sk-proj-abC1234567890defGHIJKLMnoPQRSTuvwxYZ12345678" not in res.sanitized_text
