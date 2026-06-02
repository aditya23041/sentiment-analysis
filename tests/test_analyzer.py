"""Integration tests for the SentimentAnalyzer orchestrator."""

from __future__ import annotations

import pytest

from sentiment_analysis.core.analyzer import SentimentAnalyzer
from sentiment_analysis.core.schemas import CompareResponse, SentimentLabel, SentimentResult

from .conftest import POSITIVE_TEXTS


class TestSentimentAnalyzer:
    """Tests for the main SentimentAnalyzer class."""

    def test_default_model_is_vader(self, vader_analyzer: SentimentAnalyzer):
        assert vader_analyzer.model_name == "vader"

    def test_textblob_model(self, textblob_analyzer: SentimentAnalyzer):
        assert textblob_analyzer.model_name == "textblob"

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            SentimentAnalyzer(model="nonexistent")

    def test_analyze_returns_sentiment_result(self, vader_analyzer: SentimentAnalyzer):
        result = vader_analyzer.analyze("I love this!")
        assert isinstance(result, SentimentResult)

    def test_analyze_positive(self, vader_analyzer: SentimentAnalyzer):
        result = vader_analyzer.analyze("This is absolutely fantastic!")
        assert result.polarity > 0

    def test_analyze_negative(self, vader_analyzer: SentimentAnalyzer):
        result = vader_analyzer.analyze("This is absolutely terrible!")
        assert result.polarity < 0

    def test_analyze_empty_raises(self, vader_analyzer: SentimentAnalyzer):
        with pytest.raises(ValueError, match="empty"):
            vader_analyzer.analyze("")

    def test_analyze_whitespace_raises(self, vader_analyzer: SentimentAnalyzer):
        with pytest.raises(ValueError, match="empty"):
            vader_analyzer.analyze("   ")

    def test_analyze_preserves_original_text(self, vader_analyzer: SentimentAnalyzer):
        """Result should contain the original text, not preprocessed text."""
        original = "I love this! https://example.com"
        result = vader_analyzer.analyze(original)
        assert result.text == original

    def test_batch_analysis(self, vader_analyzer: SentimentAnalyzer):
        results = vader_analyzer.analyze_batch(POSITIVE_TEXTS)
        assert len(results) == len(POSITIVE_TEXTS)
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_batch_empty_raises(self, vader_analyzer: SentimentAnalyzer):
        with pytest.raises(ValueError, match="empty"):
            vader_analyzer.analyze_batch([])


class TestModelComparison:
    """Tests for multi-model comparison."""

    def test_compare_returns_compare_response(self, vader_analyzer: SentimentAnalyzer):
        result = vader_analyzer.compare_models("I love this product!")
        assert isinstance(result, CompareResponse)
        assert result.text == "I love this product!"

    def test_compare_has_multiple_models(self, vader_analyzer: SentimentAnalyzer):
        result = vader_analyzer.compare_models("This is great!")
        assert len(result.results) >= 2  # At least vader and textblob

    def test_compare_consensus_on_clear_positive(self, vader_analyzer: SentimentAnalyzer):
        result = vader_analyzer.compare_models("I absolutely love this! Best thing ever!")
        # All models should agree on a clearly positive text
        if result.consensus:
            assert result.consensus in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)

    def test_compare_empty_raises(self, vader_analyzer: SentimentAnalyzer):
        with pytest.raises(ValueError, match="empty"):
            vader_analyzer.compare_models("")


class TestCSVAnalysis:
    """Tests for CSV file analysis."""

    def test_csv_file_analysis(self, vader_analyzer: SentimentAnalyzer, sample_csv_file):
        df = vader_analyzer.analyze_csv(str(sample_csv_file), text_column="text")
        assert len(df) == 5
        assert "polarity" in df.columns
        assert "sentiment" in df.columns
        assert "confidence" in df.columns

    def test_csv_missing_file(self, vader_analyzer: SentimentAnalyzer):
        with pytest.raises(FileNotFoundError):
            vader_analyzer.analyze_csv("nonexistent.csv")

    def test_csv_missing_column(self, vader_analyzer: SentimentAnalyzer, sample_csv_file):
        with pytest.raises(KeyError, match="not found"):
            vader_analyzer.analyze_csv(str(sample_csv_file), text_column="nonexistent")

    def test_csv_content_analysis(self, vader_analyzer: SentimentAnalyzer, sample_csv_content):
        df = vader_analyzer.analyze_csv_content(sample_csv_content, text_column="text")
        assert len(df) == 5
        assert "sentiment" in df.columns
