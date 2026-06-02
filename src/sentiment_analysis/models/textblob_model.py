"""TextBlob sentiment analysis backend."""

from __future__ import annotations

from textblob import TextBlob

from sentiment_analysis.core.schemas import SentimentResult
from sentiment_analysis.models.base import SentimentModel


class TextBlobModel(SentimentModel):
    """Sentiment analysis using TextBlob's pattern-based approach.

    TextBlob uses a lexicon-based method derived from the Pattern library.
    It provides polarity (-1 to 1) and subjectivity (0 to 1) scores.

    Best for: Simple, general-purpose text. Fast but lower accuracy.
    """

    @property
    def name(self) -> str:
        return "textblob"

    def analyze(self, text: str) -> SentimentResult:
        """Analyze text sentiment using TextBlob.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult with polarity, subjectivity, and 5-class label.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        sentiment = SentimentResult.polarity_to_label(polarity)

        # TextBlob doesn't provide a confidence score natively;
        # we use the absolute polarity as a rough proxy.
        confidence = min(abs(polarity) * 1.5, 1.0)

        return SentimentResult(
            text=text,
            polarity=round(polarity, 4),
            subjectivity=round(subjectivity, 4),
            sentiment=sentiment,
            confidence=round(confidence, 4),
            model_used=self.name,
        )
