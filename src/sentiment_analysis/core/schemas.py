"""Pydantic data models for sentiment analysis results and requests."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SentimentLabel(str, enum.Enum):
    """Five-class sentiment classification."""

    VERY_NEGATIVE = "VERY_NEGATIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    VERY_POSITIVE = "VERY_POSITIVE"


class SentimentResult(BaseModel):
    """Result of analyzing a single text."""

    text: str = Field(description="The original input text")
    polarity: float = Field(description="Polarity score from -1.0 (negative) to 1.0 (positive)")
    subjectivity: float = Field(
        default=0.0, description="Subjectivity score from 0.0 (objective) to 1.0 (subjective)"
    )
    sentiment: SentimentLabel = Field(description="Classified sentiment label")
    confidence: float = Field(
        default=0.0, description="Confidence score from 0.0 to 1.0"
    )
    model_used: str = Field(description="Name of the model that produced this result")
    is_sarcastic: bool = Field(default=False, description="True if the text is deemed sarcastic")
    sarcasm_probability: float = Field(default=0.0, description="Probability of sarcasm from 0.0 to 1.0")
    metadata: dict = Field(default_factory=dict, description="Additional context such as emotions mapping")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def polarity_to_label(polarity: float) -> SentimentLabel:
        """Convert a polarity score to a 5-class sentiment label.

        Thresholds:
            polarity <= -0.6  -> VERY_NEGATIVE
            -0.6 < polarity <= -0.1  -> NEGATIVE
            -0.1 < polarity < 0.1  -> NEUTRAL
            0.1 <= polarity < 0.6  -> POSITIVE
            polarity >= 0.6  -> VERY_POSITIVE
        """
        if polarity <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif polarity <= -0.1:
            return SentimentLabel.NEGATIVE
        elif polarity < 0.1:
            return SentimentLabel.NEUTRAL
        elif polarity < 0.6:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.VERY_POSITIVE


class AnalysisRequest(BaseModel):
    """Request to analyze a single text."""

    text: str = Field(min_length=1, max_length=10_000, description="Text to analyze")
    model: str = Field(default="vader", description="Model backend to use")


class BatchAnalysisRequest(BaseModel):
    """Request to analyze multiple texts."""

    texts: list[str] = Field(min_length=1, max_length=500, description="List of texts to analyze")
    model: str = Field(default="vader", description="Model backend to use")


class CompareRequest(BaseModel):
    """Request to compare all models on the same text."""

    text: str = Field(min_length=1, max_length=10_000, description="Text to analyze")


class AnalysisResponse(BaseModel):
    """Response containing analysis result(s)."""

    results: list[SentimentResult]
    total: int = Field(description="Total number of results")
    model_used: str = Field(description="Model that produced the results")


class CompareResponse(BaseModel):
    """Response comparing results from multiple models."""

    text: str
    results: dict[str, SentimentResult] = Field(
        description="Results keyed by model name"
    )
    consensus: SentimentLabel | None = Field(
        default=None, description="Majority sentiment if models agree"
    )


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str
    available_models: list[str]
