"""Abstract base class for sentiment analysis model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentiment_analysis.core.schemas import SentimentResult


class SentimentModel(ABC):
    """Abstract base class that all sentiment model backends must implement.

    Subclasses must implement:
        - name: A unique string identifier for the model.
        - analyze(text): Analyze a single text and return a SentimentResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this model backend."""
        ...

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze the sentiment of a single text.

        Args:
            text: The input text to analyze.

        Returns:
            A SentimentResult with polarity, subjectivity, sentiment label,
            confidence, and model name.
        """
        ...

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment for a batch of texts.

        Default implementation iterates one-by-one. Subclasses can override
        for more efficient batched inference.

        Args:
            texts: List of input texts.

        Returns:
            List of SentimentResult objects.
        """
        return [self.analyze(text) for text in texts]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model='{self.name}'>"
