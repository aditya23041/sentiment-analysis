"""VADER sentiment analysis backend."""

from __future__ import annotations

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sentiment_analysis.core.schemas import SentimentResult
from sentiment_analysis.models.base import SentimentModel

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class VaderModel(SentimentModel):
    """Sentiment analysis using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is specifically tuned for social media and short-form text. It handles
    emoticons, slang, capitalization emphasis, and degree modifiers well.

    Best for: Social media posts, reviews, short informal text.
    """

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    @property
    def name(self) -> str:
        return "vader"

    def analyze(self, text: str) -> SentimentResult:
        """Analyze text sentiment using VADER.

        VADER returns a compound score (-1 to 1) which is a normalized,
        weighted composite of lexical features. We map this to our 5-class
        sentiment system.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult with compound-derived polarity and 5-class label.
        """
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]

        # Map compound score to 5-class sentiment
        sentiment = SentimentResult.polarity_to_label(compound)

        # VADER compound confidence: use the magnitude of the compound score
        confidence = min(abs(compound) * 1.2, 1.0)

        # Compute subjectivity proxy from the ratio of non-neutral scores
        pos = scores["pos"]
        neg = scores["neg"]
        subjectivity = round(pos + neg, 4)  # Higher = more opinionated

        return SentimentResult(
            text=text,
            polarity=round(compound, 4),
            subjectivity=subjectivity,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            model_used=self.name,
        )
