"""SnapShield AI Data Module.

Provides loaders for synthetic privacy evaluation samples and demo documents.
Strictly zero real personal data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_sample_docs_dir() -> Path:
    """Returns the path to the sample_docs directory."""
    local_dir = Path(__file__).parent / "sample_docs"
    if local_dir.exists():
        return local_dir
    # Fallback to root data/sample_docs
    root_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample_docs"
    if root_dir.exists():
        return root_dir
    return local_dir


def load_sample_document(filename: str) -> str:
    """Loads text content of a specific synthetic sample document."""
    docs_dir = get_sample_docs_dir()
    target = docs_dir / filename
    if target.exists():
        return target.read_text(encoding="utf-8")
    return ""


def list_sample_documents() -> Dict[str, Dict[str, str]]:
    """Returns catalog of available synthetic demo documents for UI loading."""
    docs_dir = get_sample_docs_dir()
    catalog = {
        "Employee Onboarding Record": {
            "filename": "employee_record.txt",
            "description": "Name, Address, Aadhaar, PAN, Phone, Email, Bank Account, IFSC",
            "recommended_policy": "Personal",
        },
        "Candidate Resume (CV)": {
            "filename": "resume.txt",
            "description": "Candidate profile, contact details, work history, banking references",
            "recommended_policy": "Personal",
        },
        "Clinical Inpatient Medical Note": {
            "filename": "medical_note.txt",
            "description": "Patient info, diagnosis, attending physician, hospital, clinical observations",
            "recommended_policy": "Medical",
        },
        "Developer Infrastructure Config": {
            "filename": "developer_config.txt",
            "description": "API keys, passwords, JWT bearer tokens, database connection string, IP address",
            "recommended_policy": "Developer",
        },
    }

    # Ensure content is loaded
    for key, item in catalog.items():
        doc_path = docs_dir / item["filename"]
        if doc_path.exists():
            item["content"] = doc_path.read_text(encoding="utf-8")
        else:
            item["content"] = ""

    return catalog


def load_synthetic_samples() -> List[Dict[str, Any]]:
    """Loads synthetic privacy evaluation test samples from JSON."""
    samples_path = Path(__file__).parent / "synthetic_samples.json"
    if not samples_path.exists():
        return []
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)
