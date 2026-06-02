"""Shared test fixtures for sentiment analysis tests."""

from __future__ import annotations

import pytest

from sentiment_analysis.core.analyzer import SentimentAnalyzer
from sentiment_analysis.models.textblob_model import TextBlobModel
from sentiment_analysis.models.vader_model import VaderModel

# --- Sample Texts ---

POSITIVE_TEXTS = [
    "I love this product! It's amazing and works perfectly.",
    "Best purchase I've ever made! Highly recommend.",
    "This is absolutely wonderful, exceeded all expectations!",
]

NEGATIVE_TEXTS = [
    "This is terrible. Complete waste of money.",
    "I absolutely hate this product. Never buying again.",
    "Worst experience ever. The service was awful.",
]

NEUTRAL_TEXTS = [
    "The meeting is at 3 PM.",
    "Water boils at 100 degrees Celsius.",
    "The report was submitted on Friday.",
]

EDGE_CASE_TEXTS = [
    "",
    "   ",
    "a",
    "😊😊😊",
    "https://example.com check this out @user #trending",
    "I don't not dislike it",  # Double negation
    "A" * 5000,  # Very long text
]


@pytest.fixture
def vader_model():
    """Fresh VADER model instance."""
    return VaderModel()


@pytest.fixture
def textblob_model():
    """Fresh TextBlob model instance."""
    return TextBlobModel()


@pytest.fixture
def vader_analyzer():
    """Analyzer with VADER model."""
    return SentimentAnalyzer(model="vader")


@pytest.fixture
def textblob_analyzer():
    """Analyzer with TextBlob model."""
    return SentimentAnalyzer(model="textblob")


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return (
        "text,category\n"
        '"I love this!",positive\n'
        '"This is terrible",negative\n'
        '"The weather is okay",neutral\n'
        '"Absolutely fantastic product!",positive\n'
        '"Worst service ever",negative\n'
    )


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_content):
    """Sample CSV file on disk."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(sample_csv_content)
    return csv_file
