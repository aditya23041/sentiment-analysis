"""LLM-powered sarcasm-aware sentiment analysis backend.

Falls back to VADER automatically if the LLM API is unreachable
(e.g. Groq blocked by Cloudflare on free-tier cloud hosts).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx
from openai import OpenAI

from sentiment_analysis.core.schemas import SentimentLabel, SentimentResult
from sentiment_analysis.models.base import SentimentModel

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_SYSTEM_PROMPT = (
    "You are an advanced Contextual Nuance Agent for sentiment analysis.\n"
    "Classify the user's text into exactly one category:\n"
    "VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Watch for sarcasm, passive-aggression, and idioms.\n"
    "2. 'Oh great, another flat tire' is sarcastic → NEGATIVE.\n"
    "3. 'Crying tears of joy' → VERY_POSITIVE.\n"
    "4. Output ONLY the category name. No explanation."
)

_POLARITY_MAP = {
    SentimentLabel.VERY_POSITIVE: 0.9,
    SentimentLabel.POSITIVE: 0.5,
    SentimentLabel.NEUTRAL: 0.0,
    SentimentLabel.NEGATIVE: -0.5,
    SentimentLabel.VERY_NEGATIVE: -0.9,
}


class LLMSentimentModel(SentimentModel):
    """Sarcasm-aware LLM backend with automatic VADER fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "llama3-8b-8192",
    ) -> None:
        """Initialise the LLM client."""
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        # Auto-detect Groq keys
        if (
            self.api_key
            and self.api_key.startswith("gsk_")
            and not self.base_url
        ):
            self.base_url = "https://api.groq.com/openai/v1"

        self.model_name = model_name
        self.client: OpenAI | None = None
        self._fallback: SentimentModel | None = None

        if self.api_key:
            http_client = httpx.Client(
                transport=httpx.HTTPTransport(retries=3),
                timeout=15.0,
                headers={"User-Agent": _USER_AGENT},
            )
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                max_retries=3,
            )

    @property
    def name(self) -> str:
        return "llm"

    @property
    def is_available(self) -> bool:
        """Return True if an API key was found."""
        return self.client is not None

    def _get_fallback(self) -> SentimentModel:
        """Lazy-load VADER as a fallback model."""
        if self._fallback is None:
            from sentiment_analysis.models.vader_model import VaderModel
            self._fallback = VaderModel()
        return self._fallback

    def analyze(self, text: str) -> SentimentResult:
        """Analyse text using the LLM; fall back to VADER on failure."""
        if not self.is_available:
            logger.warning("LLM API key missing — falling back to VADER")
            return self._analyze_fallback(text)

        try:
            return self._analyze_llm(text)
        except Exception:
            logger.warning(
                "LLM API unreachable — falling back to VADER",
                exc_info=True,
            )
            return self._analyze_fallback(text)

    def _analyze_llm(self, text: str) -> SentimentResult:
        """Call the LLM API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        raw = response.choices[0].message.content.strip().upper()

        try:
            sentiment = SentimentLabel(raw)
        except ValueError:
            sentiment = SentimentLabel.NEUTRAL

        return SentimentResult(
            text=text,
            polarity=_POLARITY_MAP.get(sentiment, 0.0),
            subjectivity=0.8,
            sentiment=sentiment,
            confidence=0.95,
            model_used="llm_sarcasm",
            timestamp=datetime.now(timezone.utc),
        )

    def _analyze_fallback(self, text: str) -> SentimentResult:
        """Use VADER as fallback, tagging model_used accordingly."""
        fallback = self._get_fallback()
        result = fallback.analyze(text)
        # Tag it so the UI knows this was a fallback
        return SentimentResult(
            text=result.text,
            polarity=result.polarity,
            subjectivity=result.subjectivity,
            sentiment=result.sentiment,
            confidence=result.confidence,
            model_used="llm_fallback_vader",
            timestamp=datetime.now(timezone.utc),
        )
