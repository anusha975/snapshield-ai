"""Policy configuration parser and evaluator for SnapShield AI.

Implements:
- Strongly typed policy schemas (Pydantic).
- Invariant: Unknown sensitive categories must default to REVIEW.
- Normalization of category names (case-insensitive and format-agnostic).
- Separation of business policy rules from UI presentation.
"""

from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

class PolicyAction(str, Enum):
    """Authoritative policy enforcement actions."""
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


def normalize_category_key(key: str) -> str:
    """Normalizes category keys for robust lookup (case-insensitive, trimmed, unified spaces/underscores)."""
    return key.strip().upper().replace(" ", "_").replace("-", "_")


import logging
logger = logging.getLogger("snapshield.policies")





class ProfileRuleSet(BaseModel):
    """Rule set for a specific policy profile (e.g. Personal or Developer)."""
    description: str = ""
    rules: Dict[str, PolicyAction] = Field(default_factory=dict)
    _normalized_rules: Dict[str, PolicyAction] = {}

    def model_post_init(self, __context) -> None:
        """Build normalized key lookup index."""
        self._normalized_rules = {
            normalize_category_key(k): v for k, v in self.rules.items()
        }

    def resolve_action(self, category: str, default_action: PolicyAction = PolicyAction.REVIEW) -> PolicyAction:
        """Resolves action for a given category.

        Invariant: Unknown sensitive categories MUST default to REVIEW.
        """
        normalized = normalize_category_key(category)
        if normalized in self._normalized_rules:
            return self._normalized_rules[normalized]

        logger.info(
            "Category '%s' (normalized: '%s') not explicitly defined in profile rules. Defaulting to REVIEW.",
            category,
            normalized,
        )
        return default_action


class PoliciesConfig(BaseModel):
    """Complete policies configuration loaded from policies.json."""
    version: str = "1.0.0"
    default_action: PolicyAction = PolicyAction.REVIEW
    policies: Dict[str, ProfileRuleSet] = Field(default_factory=dict)

    @field_validator("default_action")
    @classmethod
    def validate_default_action(cls, v: PolicyAction) -> PolicyAction:
        if v != PolicyAction.REVIEW:
            logger.warning("Default action was set to %s, but invariant requires REVIEW for unknown categories.", v)
        return v

    def get_profile(self, profile_name: str) -> ProfileRuleSet:
        """Retrieves profile rule set by name (case-insensitive)."""
        for name, profile in self.policies.items():
            if name.lower() == profile_name.lower():
                return profile
        raise KeyError(f"Policy profile '{profile_name}' not found. Available profiles: {list(self.policies.keys())}")

    def evaluate_category(self, profile_name: str, category: str) -> PolicyAction:
        """Evaluates policy action for a category under the specified profile.

        Invariant: Unknown sensitive categories default to REVIEW.
        """
        try:
            profile = self.get_profile(profile_name)
            return profile.resolve_action(category, default_action=self.default_action)
        except KeyError:
            logger.warning("Requested profile '%s' does not exist. Defaulting to REVIEW.", profile_name)
            return PolicyAction.REVIEW


class PolicyConfigManager:
    """Manager for loading, caching, and querying policy configurations."""

    def __init__(self, config_file: Optional[Path | str] = None) -> None:
        if config_file is None:
            config_file = Path(__file__).parent / "policies.json"
        self.config_file = Path(config_file)
        self._config: Optional[PoliciesConfig] = None
        self.load_config()

    def load_config(self) -> PoliciesConfig:
        """Loads and parses policies from JSON file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Policy configuration file not found at: {self.config_file}")

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self._config = PoliciesConfig.model_validate(raw_data)
            logger.info("Loaded %d policy profiles from %s", len(self._config.policies), self.config_file.name)
            return self._config
        except Exception as exc:
            logger.error("Failed to load policies.json: %s", str(exc))
            raise ValueError(f"Invalid policies configuration: {exc}") from exc

    @property
    def config(self) -> PoliciesConfig:
        if self._config is None:
            self.load_config()
        return self._config  # type: ignore

    def list_profiles(self) -> List[str]:
        """Returns list of available profile names."""
        return list(self.config.policies.keys())

    def get_action_for_category(self, profile: str, category: str) -> PolicyAction:
        """Resolves the policy action for a category in a profile.

        Guarantees that unknown categories default to PolicyAction.REVIEW.
        """
        return self.config.evaluate_category(profile, category)


@lru_cache(maxsize=1)
def get_policy_manager() -> PolicyConfigManager:
    """Returns singleton instance of PolicyConfigManager."""
    return PolicyConfigManager()
