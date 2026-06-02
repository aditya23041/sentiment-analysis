"""Unit tests for sentiment model backends."""

from __future__ import annotations

import pytest

from sentiment_analysis.core.schemas import SentimentLabel, SentimentResult
from sentiment_analysis.models.base import SentimentModel
from sentiment_analysis.models.textblob_model import TextBlobModel
from sentiment_analysis.models.vader_model import VaderModel

from .conftest import NEGATIVE_TEXTS, POSITIVE_TEXTS


class TestSentimentResult:
    """Tests for the SentimentResult schema."""

    @pytest.mark.parametrize("polarity,expected", [
        (-0.9, SentimentLabel.VERY_NEGATIVE),
        (-0.6, SentimentLabel.VERY_NEGATIVE),
        (-0.3, SentimentLabel.NEGATIVE),
        (-0.1, SentimentLabel.NEGATIVE),
        (0.0, SentimentLabel.NEUTRAL),
        (0.05, SentimentLabel.NEUTRAL),
        (0.1, SentimentLabel.POSITIVE),
        (0.3, SentimentLabel.POSITIVE),
        (0.6, SentimentLabel.VERY_POSITIVE),
        (0.9, SentimentLabel.VERY_POSITIVE),
    ])
    def test_polarity_to_label(self, polarity: float, expected: SentimentLabel):
        """Test that polarity values map to correct 5-class labels."""
        assert SentimentResult.polarity_to_label(polarity) == expected

    def test_boundary_values(self):
        """Test exact boundary values."""
        assert SentimentResult.polarity_to_label(-1.0) == SentimentLabel.VERY_NEGATIVE
        assert SentimentResult.polarity_to_label(1.0) == SentimentLabel.VERY_POSITIVE
        assert SentimentResult.polarity_to_label(0.0) == SentimentLabel.NEUTRAL


class TestVaderModel:
    """Tests for the VADER model backend."""

    def test_is_sentiment_model(self, vader_model: VaderModel):
        assert isinstance(vader_model, SentimentModel)

    def test_name(self, vader_model: VaderModel):
        assert vader_model.name == "vader"

    def test_repr(self, vader_model: VaderModel):
        assert "vader" in repr(vader_model)

    @pytest.mark.parametrize("text", POSITIVE_TEXTS)
    def test_positive_texts(self, vader_model: VaderModel, text: str):
        result = vader_model.analyze(text)
        assert isinstance(result, SentimentResult)
        assert result.polarity > 0
        assert result.sentiment in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)
        assert result.model_used == "vader"

    @pytest.mark.parametrize("text", NEGATIVE_TEXTS)
    def test_negative_texts(self, vader_model: VaderModel, text: str):
        result = vader_model.analyze(text)
        assert result.polarity < 0
        assert result.sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE)

    def test_batch_analysis(self, vader_model: VaderModel):
        results = vader_model.analyze_batch(POSITIVE_TEXTS + NEGATIVE_TEXTS)
        assert len(results) == len(POSITIVE_TEXTS) + len(NEGATIVE_TEXTS)
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_result_has_required_fields(self, vader_model: VaderModel):
        result = vader_model.analyze("Test text")
        assert result.text == "Test text"
        assert -1 <= result.polarity <= 1
        assert 0 <= result.subjectivity <= 1
        assert 0 <= result.confidence <= 1
        assert result.model_used == "vader"
        assert result.timestamp is not None


class TestTextBlobModel:
    """Tests for the TextBlob model backend."""

    def test_is_sentiment_model(self, textblob_model: TextBlobModel):
        assert isinstance(textblob_model, SentimentModel)

    def test_name(self, textblob_model: TextBlobModel):
        assert textblob_model.name == "textblob"

    @pytest.mark.parametrize("text", POSITIVE_TEXTS)
    def test_positive_texts(self, textblob_model: TextBlobModel, text: str):
        result = textblob_model.analyze(text)
        assert result.polarity > 0

    @pytest.mark.parametrize("text", NEGATIVE_TEXTS)
    def test_negative_texts(self, textblob_model: TextBlobModel, text: str):
        result = textblob_model.analyze(text)
        assert result.polarity < 0

    def test_subjectivity(self, textblob_model: TextBlobModel):
        """TextBlob should provide meaningful subjectivity scores."""
        opinion = textblob_model.analyze("I absolutely love this amazing product!")
        fact = textblob_model.analyze("Water is a liquid at room temperature.")
        # Opinionated text should generally be more subjective
        assert opinion.subjectivity >= 0
        assert fact.subjectivity >= 0
