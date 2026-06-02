"""Main sentiment analyzer — orchestrates models, preprocessing, and batch processing."""

from __future__ import annotations

import io
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from sentiment_analysis.core.preprocessing import TextPreprocessor
from sentiment_analysis.core.schemas import (
    CompareResponse,
    SentimentLabel,
    SentimentResult,
)
from sentiment_analysis.models import get_model, list_models
from sentiment_analysis.models.base import SentimentModel

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """High-level sentiment analysis orchestrator.

    Wraps model backends with preprocessing, batch processing, CSV/JSON I/O,
    and multi-model comparison.

    Args:
        model: Name of the model backend to use ('textblob', 'vader', 'transformer', 'llm').
            Defaults to 'vader'.
        preprocess: Whether to preprocess text before analysis. Default True.
        preprocessor_kwargs: Additional keyword arguments for TextPreprocessor.

    Examples:
        >>> analyzer = SentimentAnalyzer(model="vader")
        >>> result = analyzer.analyze("I love this product!")
        >>> result.sentiment
        <SentimentLabel.POSITIVE: 'POSITIVE'>

        >>> results = analyzer.analyze_batch(["Great!", "Terrible.", "Okay"])
        >>> len(results)
        3
    """

    def __init__(
        self,
        model: str = "llm",
        *,
        preprocess: bool = True,
        **preprocessor_kwargs: object,
    ) -> None:
        self._model: SentimentModel = get_model(model)
        self._preprocess = preprocess
        self._preprocessor = TextPreprocessor(**preprocessor_kwargs) if preprocess else None  # type: ignore[arg-type]
        logger.info("SentimentAnalyzer initialized with model='%s'", model)

    @property
    def model_name(self) -> str:
        """Name of the active model backend."""
        return self._model.name

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.

        Args:
            text: Input text.

        Returns:
            SentimentResult with polarity, sentiment label, confidence, etc.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        processed_text = text
        if self._preprocessor:
            processed_text = self._preprocessor.process(text)
            if not processed_text:
                processed_text = text  # Fallback if preprocessing strips everything

        result = self._model.analyze(processed_text)
        # Keep original text in the result for display purposes
        result.text = text
        return result

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment for a list of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of SentimentResult objects.

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        processed_texts = []
        original_texts = []
        for text in texts:
            if not text or not text.strip():
                logger.warning("Skipping empty text in batch")
                continue
            original_texts.append(text)
            if self._preprocessor:
                processed = self._preprocessor.process(text)
                processed_texts.append(processed if processed else text)
            else:
                processed_texts.append(text)

        results = self._model.analyze_batch(processed_texts)

        # Restore original texts
        for result, original in zip(results, original_texts, strict=True):
            result.text = original

        return results

    def compare_models(self, text: str) -> CompareResponse:
        """Run the same text through all available models and compare results.

        Args:
            text: Input text to analyze across all models.

        Returns:
            CompareResponse with per-model results and consensus sentiment.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        available = list_models()
        results: dict[str, SentimentResult] = {}

        for model_name in available:
            try:
                model = get_model(model_name)
                processed_text = text
                if self._preprocessor:
                    processed_text = self._preprocessor.process(text)
                    if not processed_text:
                        processed_text = text

                result = model.analyze(processed_text)
                result.text = text
                results[model_name] = result
            except Exception:
                logger.exception("Model '%s' failed on text", model_name)

        # Determine consensus
        consensus: SentimentLabel | None = None
        if results:
            sentiment_counts = Counter(r.sentiment for r in results.values())
            most_common = sentiment_counts.most_common(1)[0]
            if most_common[1] > len(results) / 2:
                consensus = most_common[0]

        return CompareResponse(
            text=text,
            results=results,
            consensus=consensus,
        )

    def analyze_csv(
        self,
        filepath: str | Path,
        text_column: str = "text",
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Analyze sentiment from a CSV file.

        Args:
            filepath: Path to CSV file.
            text_column: Name of the column containing text data.
            encoding: File encoding. Default 'utf-8'.

        Returns:
            DataFrame with original data plus sentiment analysis columns.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            KeyError: If the text column is not found.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, encoding=encoding)

        if text_column not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            raise KeyError(
                f"Column '{text_column}' not found. Available columns: {available_cols}"
            )

        # Drop rows with missing text
        valid_mask = df[text_column].notna() & (df[text_column].str.strip() != "")
        valid_df = df[valid_mask].copy()

        if valid_df.empty:
            logger.warning("No valid text found in column '%s'", text_column)
            return valid_df

        texts = valid_df[text_column].tolist()
        results = self.analyze_batch(texts)

        # Add result columns
        valid_df["polarity"] = [r.polarity for r in results]
        valid_df["subjectivity"] = [r.subjectivity for r in results]
        valid_df["sentiment"] = [r.sentiment.value for r in results]
        valid_df["confidence"] = [r.confidence for r in results]
        valid_df["model_used"] = [r.model_used for r in results]

        return valid_df

    def analyze_csv_content(
        self,
        content: str | bytes,
        text_column: str = "text",
    ) -> pd.DataFrame:
        """Analyze sentiment from CSV content (e.g. uploaded file).

        Args:
            content: CSV content as string or bytes.
            text_column: Name of the column containing text data.

        Returns:
            DataFrame with sentiment analysis columns appended.
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        df = pd.read_csv(io.StringIO(content))

        if text_column not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            raise KeyError(
                f"Column '{text_column}' not found. Available columns: {available_cols}"
            )

        valid_mask = df[text_column].notna() & (df[text_column].str.strip() != "")
        valid_df = df[valid_mask].copy()

        if valid_df.empty:
            return valid_df

        texts = valid_df[text_column].tolist()
        results = self.analyze_batch(texts)

        valid_df["polarity"] = [r.polarity for r in results]
        valid_df["subjectivity"] = [r.subjectivity for r in results]
        valid_df["sentiment"] = [r.sentiment.value for r in results]
        valid_df["confidence"] = [r.confidence for r in results]
        valid_df["model_used"] = [r.model_used for r in results]

        return valid_df
