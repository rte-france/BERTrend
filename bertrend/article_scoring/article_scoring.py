#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    ConfigDict,
    computed_field,
)
from typing import Optional, List, Dict, Any
from enum import Enum


class QualityLevel(str, Enum):
    """Quality level categories based on score ranges"""

    POOR = "Poor"  # (0.0-0.25)
    FAIR = "Fair"  # (0.25-0.50)
    AVERAGE = "Average"  # (0.50-0.65)
    GOOD = "Good"  # (0.65-0.8)
    EXCELLENT = "Excellent"  # (0.8-1.0)


class CriteriaScores(BaseModel):
    """Individual scores for each quality criterion"""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    depth_of_reporting: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the article provides background, context, explanations, and connects to broader trends",
    )
    originality_and_exclusivity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Unique information, exclusive insights, original reporting, or fresh perspectives",
    )
    source_quality_and_transparency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality and transparency of sources, credibility, independence, and conflict disclosure",
    )
    accuracy_and_fact_checking_rigor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision, verifiability, appropriate evidence strength, and error handling",
    )
    clarity_and_accessibility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Clear communication, logical hierarchy, and accessibility for target audience",
    )
    balance_and_fairness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fair presentation of viewpoints, absence of bias, acknowledgment of uncertainties",
    )
    narrative_and_engagement: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reader engagement, storytelling effectiveness, and balance with journalistic integrity",
    )
    timeliness_and_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Currency of information, relevance to current context, and appropriate timing",
    )
    ethical_considerations_and_sensitivity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ethical reporting standards, sensitivity to affected populations, avoiding sensationalism",
    )
    rte_relevance_and_strategic_impact: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance to RTE operations, energy transmission, grid management, and strategic planning",
    )

    @classmethod
    def get_criterion_names(cls) -> List[str]:
        """Get list of all criterion names"""
        return list(cls.model_fields.keys())

    def to_dict(self) -> Dict[str, float]:
        """Convert scores to dictionary"""
        return {name: getattr(self, name) for name in self.get_criterion_names()}


class WeightConfig(BaseModel):
    """Configurable weights for each criterion"""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    depth_of_reporting: float = Field(0.15, ge=0.0, le=1.0)
    originality_and_exclusivity: float = Field(0.10, ge=0.0, le=1.0)
    source_quality_and_transparency: float = Field(0.15, ge=0.0, le=1.0)
    accuracy_and_fact_checking_rigor: float = Field(0.15, ge=0.0, le=1.0)
    clarity_and_accessibility: float = Field(0.10, ge=0.0, le=1.0)
    balance_and_fairness: float = Field(0.10, ge=0.0, le=1.0)
    narrative_and_engagement: float = Field(0.10, ge=0.0, le=1.0)
    timeliness_and_relevance: float = Field(0.05, ge=0.0, le=1.0)
    ethical_considerations_and_sensitivity: float = Field(0.05, ge=0.0, le=1.0)
    rte_relevance_and_strategic_impact: float = Field(0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_validator(self) -> "WeightConfig":
        """Ensure all weights sum to less than the number of criteria"""
        total = sum(
            getattr(self, name) for name in CriteriaScores.get_criterion_names()
        )
        if not total <= len(CriteriaScores.get_criterion_names()):
            raise ValueError(
                f"Weights must not exceed {len(CriteriaScores.get_criterion_names())}, got {total:.6f}"
            )
        return self

    def to_dict(self) -> Dict[str, float]:
        """Convert weights to dictionary"""
        return {
            name: getattr(self, name) for name in CriteriaScores.get_criterion_names()
        }


class ArticleScore(BaseModel):
    """Complete article scoring with individual scores and final result"""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Required fields
    scores: CriteriaScores

    # Optional configuration
    weights: WeightConfig = Field(default_factory=WeightConfig)

    # Assessment metadata
    assessment_summary: Optional[str] = Field(
        None, description="Brief summary of strengths and weaknesses"
    )

    @computed_field
    @property
    def final_score(self) -> float:
        """Calculate weighted final score"""
        score_dict = self.scores.to_dict()
        weight_dict = self.weights.to_dict()

        weighted_sum = sum(
            score_dict[name] * weight_dict[name] for name in score_dict.keys()
        )
        return round(weighted_sum, 3) / len(score_dict)

    @computed_field
    @property
    def quality_level(self) -> QualityLevel:
        """Determine quality level based on final score"""
        score = self.final_score
        if 0.0 <= score < 0.25:
            return QualityLevel.POOR
        elif 0.25 <= score < 0.50:
            return QualityLevel.FAIR
        elif 0.5 <= score < 0.65:
            return QualityLevel.AVERAGE
        elif 0.65 <= score < 0.8:
            return QualityLevel.GOOD
        else:  # 0.8 <= score <= 1.0
            return QualityLevel.EXCELLENT

    def get_detailed_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed score breakdown with weights and contributions"""
        score_dict = self.scores.to_dict()
        weight_dict = self.weights.to_dict()

        return {
            criterion: {
                "score": score_dict[criterion],
                "weight": weight_dict[criterion],
                "weighted_contribution": round(
                    score_dict[criterion] * weight_dict[criterion], 3
                ),
            }
            for criterion in score_dict.keys()
        }

    def get_top_strengths(self, n: int = 3) -> List[tuple[str, float]]:
        """Get top N scoring criteria"""
        score_dict = self.scores.to_dict()
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n]

    def get_top_weaknesses(self, n: int = 3) -> List[tuple[str, float]]:
        """Get bottom N scoring criteria"""
        score_dict = self.scores.to_dict()
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1])
        return sorted_scores[:n]

    def to_report(self, include_breakdown: bool = True) -> str:
        """Generate a formatted assessment report"""
        report_lines = [
            "ARTICLE QUALITY ASSESSMENT REPORT",
            "=" * 50,
            "",
            f"FINAL SCORE: {self.final_score}/1.0 ({self.quality_level.value})",
            "",
        ]

        if include_breakdown:
            report_lines.append("INDIVIDUAL SCORES:")
            breakdown = self.get_detailed_breakdown()
            for i, (criterion, data) in enumerate(breakdown.items(), 1):
                criterion_name = criterion.replace("_", " ").title()
                report_lines.append(
                    f"{i:2d}. {criterion_name}: {data['score']:.2f}/1.0 "
                    f"(Weight: {data['weight']:.0%}, Contribution: {data['weighted_contribution']:.3f})"
                )
            report_lines.append("")

        # Add strengths and weaknesses
        strengths = self.get_top_strengths(3)
        weaknesses = self.get_top_weaknesses(3)

        report_lines.extend(
            [
                "TOP STRENGTHS:",
                *[
                    f"• {name.replace('_', ' ').title()}: {score:.2f}"
                    for name, score in strengths
                ],
                "",
                "AREAS FOR IMPROVEMENT:",
                *[
                    f"• {name.replace('_', ' ').title()}: {score:.2f}"
                    for name, score in weaknesses
                ],
                "",
            ]
        )

        if self.assessment_summary:
            report_lines.extend(
                [
                    "OVERALL ASSESSMENT:",
                    self.assessment_summary,
                    "",
                ]
            )

        return "\n".join(report_lines)

    def export_to_dict(self) -> Dict[str, Any]:
        """Export complete assessment to dictionary for serialization"""
        return {
            "scores": self.scores.model_dump(),
            "weights": self.weights.model_dump(),
            "final_score": self.final_score,
            "quality_level": self.quality_level.value,
            "assessment_summary": self.assessment_summary,
            "detailed_breakdown": self.get_detailed_breakdown(),
        }
