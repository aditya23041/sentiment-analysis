"""HuggingFace Transformers sentiment analysis backend (optional).

Requires: pip install sentiment-analysis[transformers]
"""

from __future__ import annotations

import logging
from typing import Any

from sentiment_analysis.core.schemas import SentimentResult
from sentiment_analysis.models.base import SentimentModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class TransformerModel(SentimentModel):
    """Sentiment analysis using HuggingFace Transformers pipeline.

    Uses a pre-trained transformer model (default: DistilBERT fine-tuned on SST-2)
    for high-accuracy sentiment classification. The model is loaded lazily on first use.

    Best for: Maximum accuracy. Requires more memory and compute (GPU recommended).

    Args:
        model_name: HuggingFace model identifier. Defaults to DistilBERT-SST2.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, **kwargs: Any) -> None:
        self._model_name = model_name
        self._pipeline: Any = None

    @property
    def name(self) -> str:
        return "transformer"

    def _load_pipeline(self) -> Any:
        """Lazily load the transformer pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import-untyped]

                logger.info("Loading transformer model: %s", self._model_name)
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self._model_name,
                    truncation=True,
                    max_length=512,
                )
                logger.info("Transformer model loaded successfully")
            except ImportError as err:
                raise ImportError(
                    "Transformers dependencies not installed. "
                    "Install with: pip install sentiment-analysis[transformers]"
                ) from err
        return self._pipeline

    def analyze(self, text: str) -> SentimentResult:
        """Analyze text using transformer model.

        Maps the model's POSITIVE/NEGATIVE labels to our 5-class system
        using the confidence score to determine intensity.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult with transformer-derived sentiment and confidence.
        """
        pipe = self._load_pipeline()
        result = pipe(text)[0]

        label: str = result["label"]
        score: float = result["score"]

        # Convert 2-class transformer output to polarity
        polarity = score if label.upper() == "POSITIVE" else -score

        # Map to 5-class with the transformer's own confidence
        sentiment = SentimentResult.polarity_to_label(polarity)

        return SentimentResult(
            text=text,
            polarity=round(polarity, 4),
            subjectivity=0.0,  # Transformers don't provide subjectivity
            sentiment=sentiment,
            confidence=round(score, 4),
            model_used=self.name,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Batch analyze using transformer pipeline (much faster than one-by-one).

        Args:
            texts: List of input texts.

        Returns:
            List of SentimentResult objects.
        """
        pipe = self._load_pipeline()
        raw_results = pipe(texts)

        results = []
        for text, raw in zip(texts, raw_results, strict=True):
            label = raw["label"]
            score = raw["score"]

            polarity = score if label.upper() == "POSITIVE" else -score

            sentiment = SentimentResult.polarity_to_label(polarity)

            results.append(
                SentimentResult(
                    text=text,
                    polarity=round(polarity, 4),
                    subjectivity=0.0,
                    sentiment=sentiment,
                    confidence=round(score, 4),
                    model_used=self.name,
                )
            )
        return results
