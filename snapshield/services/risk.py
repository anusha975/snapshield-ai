"""Risk Assessment Service for SnapShield AI.

Calculates composite risk scores based on detected findings, category weights,
and confidence metrics.
"""

from __future__ import annotations

from typing import Dict, List
from snapshield.core.types import Finding, RiskAssessment, RiskLevel


DEFAULT_RISK_WEIGHTS: Dict[str, float] = {
    "CREDENTIALS": 40.0,
    "PAYMENT_CARD": 35.0,
    "GOVERNMENT_ID": 30.0,
    "HEALTH_INFO": 25.0,
    "INTERNAL_INFRA": 20.0,
    "CONTACT_INFO": 10.0,
    "CUSTOM": 15.0,
}


class RiskAssessmentEngine:
    """Evaluates the security and privacy risk of detected findings."""

    def __init__(self, risk_weights: Dict[str, float] | None = None) -> None:
        self.risk_weights = risk_weights or DEFAULT_RISK_WEIGHTS

    def evaluate(self, findings: List[Finding]) -> RiskAssessment:
        """Computes a normalized risk score (0-100) and risk level tier."""
        if not findings:
            return RiskAssessment(
                total_score=0.0,
                level=RiskLevel.LOW,
                findings_count=0,
                category_breakdown={},
                summary_reasons=["No sensitive entities detected."],
            )

        raw_score = 0.0
        category_breakdown: Dict[str, int] = {}
        summary_reasons: List[str] = []

        for f in findings:
            cat_name = f.entity_category.value
            weight = self.risk_weights.get(cat_name, 15.0)
            # Factor in confidence
            contribution = weight * f.confidence_score
            raw_score += contribution
            category_breakdown[cat_name] = category_breakdown.get(cat_name, 0) + 1

        # Normalize score to 100 max
        total_score = min(100.0, round(raw_score, 1))

        if total_score >= 70.0:
            level = RiskLevel.CRITICAL
        elif total_score >= 45.0:
            level = RiskLevel.HIGH
        elif total_score >= 20.0:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        for cat, count in category_breakdown.items():
            summary_reasons.append(f"Detected {count} instance(s) in category '{cat}'.")

        return RiskAssessment(
            total_score=total_score,
            level=level,
            findings_count=len(findings),
            category_breakdown=category_breakdown,
            summary_reasons=summary_reasons,
        )
