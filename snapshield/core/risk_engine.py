"""Privacy risk evaluation engine for SnapShield AI.

Features:
- Calculates a normalized 0–100 composite privacy risk score.
- Categorizes risk into 4 tiers:
  - LOW
  - MEDIUM
  - HIGH
  - CRITICAL
- Fully configurable category weights and threshold boundaries.
- Potential severity alignment:
  - Credentials -> CRITICAL
  - Government IDs -> HIGH
  - Financial -> HIGH
  - Medical -> HIGH
  - Phone -> MEDIUM
  - Email -> MEDIUM
  - Organization -> LOW
- Transparent, explainable contributor breakdown without magic numbers:
  e.g.
  Risk Score: 82
  HIGH
  Contributors:
  Government ID +30
  Financial +25
  Medical +20
  Phone +7
- Strictly enforces privacy invariant: Never exposes raw values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from snapshield.config.policies import normalize_category_key
from snapshield.core.logger import logger
from snapshield.core.models import Detection, RiskAssessment, RiskLevel, SeverityLevel


# =====================================================================
# Default Configuration Constants (Transparent & Documented)
# =====================================================================

# Documented base weights corresponding to data sensitivity and regulatory impact.
# Government IDs (30), Financial (25), Medical (20), Phone (7), etc.
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "Credentials": 40.0,
    "Government ID": 30.0,
    "Financial": 25.0,
    "Medical": 20.0,
    "Confidential Business": 20.0,
    "Private Location": 15.0,
    "Address": 10.0,
    "Person": 8.0,
    "Phone": 7.0,
    "Email": 5.0,
    "Internal Infrastructure": 5.0,
    "Organization": 3.0,
    "Unknown Sensitive": 10.0,
}

# Potential qualitative severity per risk category
DEFAULT_CATEGORY_SEVERITY: Dict[str, SeverityLevel] = {
    "Credentials": SeverityLevel.CRITICAL,
    "Government ID": SeverityLevel.HIGH,
    "Financial": SeverityLevel.HIGH,
    "Medical": SeverityLevel.HIGH,
    "Confidential Business": SeverityLevel.HIGH,
    "Private Location": SeverityLevel.HIGH,
    "Address": SeverityLevel.MEDIUM,
    "Person": SeverityLevel.MEDIUM,
    "Phone": SeverityLevel.MEDIUM,
    "Email": SeverityLevel.MEDIUM,
    "Internal Infrastructure": SeverityLevel.LOW,
    "Organization": SeverityLevel.LOW,
    "Unknown Sensitive": SeverityLevel.MEDIUM,
}

# Thresholds defining the 4 qualitative risk score tiers
DEFAULT_RISK_THRESHOLDS: Dict[str, float] = {
    "CRITICAL": 85.0,  # Score in [85.0, 100.0] -> CRITICAL
    "HIGH": 60.0,      # Score in [60.0, 85.0)  -> HIGH (e.g. 82 -> HIGH)
    "MEDIUM": 25.0,    # Score in [25.0, 60.0)  -> MEDIUM
    "LOW": 0.0,        # Score in [0.0, 25.0)   -> LOW
}

# Mapping normalized entity types to canonical risk categories
ENTITY_TO_RISK_CATEGORY: Dict[str, str] = {
    # Credentials -> CRITICAL
    "API_KEY": "Credentials",
    "PASSWORD": "Credentials",
    "JWT": "Credentials",
    "AWS_ACCESS_KEY": "Credentials",
    "AWS_SECRET_KEY": "Credentials",
    "BEARER_TOKEN": "Credentials",
    "CREDENTIAL_URL": "Credentials",
    "PRIVATE_KEY": "Credentials",
    "DATABASE_URI": "Credentials",
    "CREDENTIALS": "Credentials",
    # Government IDs -> HIGH
    "AADHAAR": "Government ID",
    "PAN": "Government ID",
    "GOVERNMENT_ID": "Government ID",
    "SSN": "Government ID",
    "PASSPORT": "Government ID",
    # Financial -> HIGH
    "BANK_ACCOUNT": "Financial",
    "PAYMENT_CARD": "Financial",
    "IFSC": "Financial",
    "FINANCIAL": "Financial",
    "CREDIT_CARD": "Financial",
    # Medical -> HIGH
    "MEDICAL_INFORMATION": "Medical",
    "PATIENT_IDENTIFIER": "Medical",
    "HEALTH_INFO": "Medical",
    "MEDICAL": "Medical",
    # Confidential Business -> HIGH
    "CONFIDENTIAL_BUSINESS_INFORMATION": "Confidential Business",
    "CONFIDENTIAL_BUSINESS": "Confidential Business",
    # Private Location -> HIGH
    "PRIVATE_LOCATION": "Private Location",
    # Address -> MEDIUM
    "ADDRESS": "Address",
    # Person -> MEDIUM
    "PERSON": "Person",
    # Phone -> MEDIUM
    "PHONE": "Phone",
    # Email -> MEDIUM
    "EMAIL": "Email",
    # Infrastructure -> LOW
    "IP_ADDRESS": "Internal Infrastructure",
    "INTERNAL_INFRA": "Internal Infrastructure",
    # Organization -> LOW
    "ORGANIZATION": "Organization",
}


# =====================================================================
# Configuration Data Transfer Object
# =====================================================================

@dataclass
class RiskEngineConfig:
    """Configurable weights, decay factors, and thresholds for risk assessment."""
    category_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_WEIGHTS)
    )
    category_severity: Dict[str, SeverityLevel] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_SEVERITY)
    )
    thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RISK_THRESHOLDS)
    )
    diminishing_returns_factor: float = 0.5  # Weight scaling for repeat findings in same category
    max_category_points: float = 50.0        # Maximum points any single category can contribute
    auto_escalate_critical: bool = True     # Elevate tier to CRITICAL if credentials / critical entities found


# =====================================================================
# Result Domain Models
# =====================================================================

@dataclass
class RiskContributor:
    """Represents a specific category's contribution to the composite risk score.

    Invariant: Never contains or leaks raw sensitive content.
    """
    category: str
    points: float
    count: int
    severity: SeverityLevel
    explanation: str

    def display(self) -> str:
        """Returns clean formatted contributor string: e.g. 'Government ID +30'."""
        pts_str = f"{int(self.points)}" if self.points.is_integer() else f"{self.points:.1f}"
        return f"{self.category} +{pts_str}"

    def __str__(self) -> str:
        return self.display()


@dataclass
class RiskReport:
    """Complete evaluation report returned by RiskEngine.

    Invariants:
    - score is bounded [0.0, 100.0]
    - level is one of LOW, MEDIUM, HIGH, CRITICAL
    - contributors list is sorted by points descending
    - zero raw values exposed
    """
    score: float
    level: RiskLevel
    contributors: List[RiskContributor]
    findings_count: int = 0
    is_blocked: bool = False

    @property
    def formatted_contributors(self) -> List[str]:
        """Returns list of contributor display strings: ['Government ID +30', 'Financial +25', ...]."""
        return [c.display() for c in self.contributors]

    def summary(self) -> str:
        """Renders standard multi-line summary as specified:

        Risk Score: 82
        HIGH

        Contributors:
        Government ID +30
        Financial +25
        Medical +20
        Phone +7
        """
        score_str = f"{int(self.score)}" if self.score.is_integer() else f"{self.score:.1f}"
        lines = [
            f"Risk Score: {score_str}",
            f"{self.level.value}",
            "",
            "Contributors:",
        ]
        if self.contributors:
            for c in self.contributors:
                lines.append(c.display())
        else:
            lines.append("None (No sensitive entities detected)")

        return "\n".join(lines)

    def to_risk_assessment(self) -> RiskAssessment:
        """Converts to core domain RiskAssessment model for pipeline compatibility."""
        return RiskAssessment(
            score=self.score,
            level=self.level,
            findings_count=self.findings_count,
            category_breakdown={c.category: c.count for c in self.contributors},
            reasons=self.formatted_contributors,
            is_blocked=self.is_blocked,
        )


# =====================================================================
# Risk Assessment Engine
# =====================================================================

class RiskEngine:
    """Evaluates privacy and security risk from detected findings.

    Features:
    - Bounded 0–100 score calculation.
    - Transparent category weights without unexplained hardcoding.
    - Tier classification: LOW, MEDIUM, HIGH, CRITICAL.
    - Strict privacy guarantee: Never exposes raw sensitive values.
    """

    def __init__(self, config: Optional[RiskEngineConfig] = None) -> None:
        self.config = config or RiskEngineConfig()

    def resolve_category(self, raw_entity_type: str) -> str:
        """Maps an entity type string to canonical risk category."""
        normalized = normalize_category_key(raw_entity_type)
        if normalized in ENTITY_TO_RISK_CATEGORY:
            return ENTITY_TO_RISK_CATEGORY[normalized]
        # Return title-cased fallback if not explicitly registered
        return raw_entity_type.replace("_", " ").title()

    def evaluate(
        self,
        detections: List[Union[Detection, Any]],
    ) -> RiskReport:
        """Evaluates detected findings and produces a strongly typed RiskReport.

        Args:
            detections: List of Detection domain models (or legacy Finding objects).

        Returns:
            RiskReport with score, level, and contributors.
        """
        if not detections:
            return RiskReport(
                score=0.0,
                level=RiskLevel.LOW,
                contributors=[],
                findings_count=0,
                is_blocked=False,
            )

        # 1. Group findings by canonical risk category
        grouped_findings: Dict[str, List[Any]] = {}
        has_critical_finding = False

        for item in detections:
            # Extract entity_type and confidence safely
            entity_type = getattr(item, "entity_type", None) or getattr(item, "entity_category", "Unknown")
            if hasattr(entity_type, "value"):
                entity_type = entity_type.value

            cat = self.resolve_category(str(entity_type))
            grouped_findings.setdefault(cat, []).append(item)

            # Check if finding severity is CRITICAL
            sev = getattr(item, "severity", None)
            if sev == SeverityLevel.CRITICAL or sev == "CRITICAL":
                has_critical_finding = True

        contributors: List[RiskContributor] = []
        raw_total_score = 0.0

        # 2. Calculate points per category
        for cat_name, items in grouped_findings.items():
            base_weight = self.config.category_weights.get(
                cat_name, self.config.category_weights.get("Unknown Sensitive", 10.0)
            )
            cat_severity = self.config.category_severity.get(
                cat_name, SeverityLevel.MEDIUM
            )

            # Sort items by confidence descending
            sorted_items = sorted(
                items,
                key=lambda x: getattr(x, "confidence", None) or getattr(x, "confidence_score", 1.0),
                reverse=True,
            )

            cat_points = 0.0
            for idx, finding in enumerate(sorted_items):
                conf = float(getattr(finding, "confidence", None) or getattr(finding, "confidence_score", 1.0))
                # First finding contributes base_weight * confidence
                # Subsequent findings contribute with diminishing returns factor
                factor = 1.0 if idx == 0 else self.config.diminishing_returns_factor
                item_points = base_weight * conf * factor
                cat_points += item_points

            # Cap category points at configured limit
            final_cat_points = min(cat_points, self.config.max_category_points)
            final_cat_points = round(final_cat_points, 1)

            raw_total_score += final_cat_points

            contributors.append(
                RiskContributor(
                    category=cat_name,
                    points=final_cat_points,
                    count=len(items),
                    severity=cat_severity,
                    explanation=f"{len(items)} occurrence(s) contributing +{final_cat_points:.1f} points.",
                )
            )

        # 3. Sort contributors by points descending
        contributors.sort(key=lambda c: c.points, reverse=True)

        # 4. Normalize overall score to 0–100
        score = min(100.0, max(0.0, round(raw_total_score, 1)))

        # 5. Determine qualitative risk tier
        thresh = self.config.thresholds
        crit_thresh = thresh.get("CRITICAL", 85.0)
        high_thresh = thresh.get("HIGH", 60.0)
        med_thresh = thresh.get("MEDIUM", 25.0)

        if score >= crit_thresh:
            level = RiskLevel.CRITICAL
        elif score >= high_thresh:
            level = RiskLevel.HIGH
        elif score >= med_thresh:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        # Escalation rule: If credentials / critical findings exist, tier is elevated to at least HIGH/CRITICAL
        if has_critical_finding and self.config.auto_escalate_critical:
            if level in (RiskLevel.LOW, RiskLevel.MEDIUM):
                level = RiskLevel.HIGH

        logger.info(
            "RiskEngine evaluated %d detection(s): Score=%.1f, Level=%s, Contributors=%d",
            len(detections),
            score,
            level.value,
            len(contributors),
        )

        return RiskReport(
            score=score,
            level=level,
            contributors=contributors,
            findings_count=len(detections),
            is_blocked=(level == RiskLevel.CRITICAL),
        )


# Module-level convenience function
def calculate_risk(
    detections: List[Union[Detection, Any]],
    config: Optional[RiskEngineConfig] = None,
) -> RiskReport:
    """Convenience function calculating risk report directly from detections."""
    engine = RiskEngine(config=config)
    return engine.evaluate(detections)
