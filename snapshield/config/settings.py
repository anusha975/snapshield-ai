"""Application configuration and settings management for SnapShield AI.

Ensures all configuration is loaded safely from environment variables and defaults,
with rigorous validation and zero hardcoding of secrets or credentials.

Supported Environment Variables:
- LLM_API_KEY: Downstream LLM authentication credential
- LLM_PROVIDER: Provider name (e.g. openai, azure, anthropic, local)
- LLM_MODEL: Model identifier (e.g. gpt-4o-mini)
- MODEL_DIRECTORY: Filesystem path where local ONNX models reside
- MAX_FILE_SIZE_MB: Maximum allowable document/image upload size in megabytes
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging

logger = logging.getLogger("snapshield.settings")


class ConfigurationValidationError(Exception):
    """Raised when environment or runtime settings fail validation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}




class SnapShieldSettings(BaseSettings):
    """Strongly typed application configuration settings with validation."""

    # Environment
    app_env: Literal["development", "testing", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )
    app_debug: bool = Field(default=True, validation_alias="APP_DEBUG")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Base Paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    # Local Model Storage (MODEL_DIRECTORY)
    model_directory: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "models",
        validation_alias="MODEL_DIRECTORY",
        description="Path to the directory housing local ONNX models.",
    )

    # File Size Limitations (MAX_FILE_SIZE_MB)
    max_file_size_mb: float = Field(
        default=25.0,
        validation_alias="MAX_FILE_SIZE_MB",
        gt=0.0,
        le=500.0,
        description="Maximum upload size in megabytes (0.1 to 500 MB).",
    )

    # Privacy Policy & Risk Control
    policy_profile: str = Field(
        default="Personal",
        validation_alias="SNAPSHIELD_POLICY_PROFILE",
        description="Active privacy profile (Personal, Developer, etc.).",
    )
    max_risk_score: float = Field(default=40.0, ge=0.0, le=100.0)
    default_policy_action: str = Field(
        default="REVIEW",
        validation_alias="SNAPSHIELD_DEFAULT_ACTION",
        description="Fallback policy action for unknown sensitive categories (strictly REVIEW).",
    )

    # Hardware & Snapdragon / NPU Acceleration
    npu_acceleration_preference: Literal["AUTO", "QNN_NPU", "DIRECTML", "CPU"] = Field(
        default="AUTO",
        validation_alias="NPU_ACCELERATION_PREFERENCE",
    )
    verify_npu_runtime: bool = Field(default=True, validation_alias="VERIFY_NPU_RUNTIME")

    # OCR Configuration
    ocr_engine: str = Field(default="tesseract_fallback", validation_alias="OCR_ENGINE")
    ocr_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        validation_alias="OCR_CONFIDENCE_THRESHOLD",
    )

    # External LLM Integration
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias="LLM_API_KEY",
        description="API key for optional external LLM service. Never logged raw.",
    )
    llm_provider: str = Field(
        default="openai",
        validation_alias="LLM_PROVIDER",
        description="External LLM provider identifier (e.g. openai, azure, anthropic, local).",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="LLM_MODEL",
        description="Downstream AI model identifier.",
    )
    external_llm_enabled: bool = Field(
        default=False,
        validation_alias="EXTERNAL_LLM_ENABLED",
    )
    external_llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="EXTERNAL_LLM_BASE_URL",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Field Validators ---

    @field_validator("model_directory", mode="before")
    @classmethod
    def validate_model_directory(cls, v: str | Path | None) -> Path:
        """Ensures model directory path is resolved and validated."""
        if v is None or str(v).strip() == "":
            resolved = Path(__file__).resolve().parent.parent / "models"
        else:
            resolved = Path(v).resolve()
        return resolved

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_file_size_bounds(cls, v: float) -> float:
        """Enforces positive, bounded max file size limits."""
        if v <= 0.0:
            raise ValueError("MAX_FILE_SIZE_MB must be strictly greater than 0.")
        if v > 500.0:
            raise ValueError("MAX_FILE_SIZE_MB cannot exceed 500 MB.")
        return round(float(v), 2)

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Ensures LLM provider is non-empty and stripped."""
        stripped = v.strip().lower()
        if not stripped:
            raise ValueError("LLM_PROVIDER cannot be empty.")
        return stripped

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        """Ensures LLM model is non-empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("LLM_MODEL cannot be empty.")
        return stripped

    @field_validator("default_policy_action")
    @classmethod
    def validate_default_action(cls, v: str) -> str:
        """Enforces that unknown sensitive categories default to REVIEW."""
        stripped = v.strip().upper()
        if stripped not in ["ALLOW", "REDACT", "BLOCK", "REVIEW"]:
            raise ValueError(f"Invalid default policy action: {v}. Must be ALLOW, REDACT, BLOCK, or REVIEW.")
        return stripped

    # --- Domain Helper Validation Methods ---

    @property
    def max_file_size_bytes(self) -> int:
        """Returns the maximum file size in bytes."""
        return int(self.max_file_size_mb * 1024 * 1024)

    def validate_file_size(self, size_bytes: int) -> None:
        """Validates an incoming file's byte size against configured limits.

        Raises:
            ConfigurationValidationError: If file size exceeds the limit.
        """
        if size_bytes < 0:
            raise ConfigurationValidationError("File size cannot be negative.")
        if size_bytes > self.max_file_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            raise ConfigurationValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {self.max_file_size_mb} MB.",
                details={"file_size_bytes": size_bytes, "max_allowed_bytes": self.max_file_size_bytes}
            )

    def validate_llm_configuration(self) -> None:
        """Validates that external LLM configuration is ready for transmission if enabled.

        Raises:
            ConfigurationValidationError: If external LLM is enabled but API key is missing.
        """
        if self.external_llm_enabled:
            if not self.llm_api_key or not self.llm_api_key.strip():
                raise ConfigurationValidationError(
                    "EXTERNAL_LLM_ENABLED is True, but LLM_API_KEY is not configured.",
                    details={"provider": self.llm_provider, "model": self.llm_model}
                )

    def has_llm_credentials(self) -> bool:
        """Returns True if LLM credentials are set without exposing the key value."""
        return bool(self.llm_api_key and self.llm_api_key.strip())

    def ensure_model_directory_exists(self) -> Path:
        """Ensures the configured model directory exists on the filesystem."""
        self.model_directory.mkdir(parents=True, exist_ok=True)
        return self.model_directory


@lru_cache(maxsize=1)
def get_settings() -> SnapShieldSettings:
    """Retrieve cached application settings instance."""
    return SnapShieldSettings()
