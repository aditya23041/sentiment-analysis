import os
import httpx
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from sentiment_analysis.core.schemas import SentimentLabel, SentimentResult
from sentiment_analysis.models.base import SentimentModel


class LLMSentimentModel(SentimentModel):
    """Sarcasm-aware LLM backend using OpenAI-compatible APIs (Groq/OpenAI)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: str = "llama3-8b-8192"):
        """Initialize the LLM client. Defaults to Groq via base_url, or standard OpenAI if not provided."""
        # Try to pull API keys from environment if not explicitly provided
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        
        # If the user provided a Groq API key but no base URL, default to Groq's endpoint
        if self.api_key and self.api_key.startswith("gsk_") and not self.base_url:
            self.base_url = "https://api.groq.com/openai/v1"
            
        self.model_name = model_name
        self.client = None
        
        if self.api_key:
            http_client = httpx.Client(
                transport=httpx.HTTPTransport(retries=3),
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
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
        """Returns True if an API key was successfully found."""
        return self.client is not None

    def analyze(self, text: str) -> SentimentResult:
        if not self.is_available:
            raise ValueError(
                "API Key missing. Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
            )

        system_prompt = (
            "You are an advanced Contextual Nuance Agent designed for sentiment analysis.\n"
            "Your job is to read the user's text and classify it into exactly one of these five categories:\n"
            "VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Pay close attention to sarcasm, passive-aggression, paradoxes, and idioms.\n"
            "2. If someone says 'Oh great, another flat tire', they are being sarcastic. The sentiment is NEGATIVE.\n"
            "3. If someone says 'I was crying tears of joy', the sentiment is VERY_POSITIVE.\n"
            "4. Output ONLY the category name and nothing else. No explanation."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            result_text = response.choices[0].message.content.strip().upper()
            
            # Map LLM output to enum, fallback to NEUTRAL
            try:
                sentiment = SentimentLabel(result_text)
            except ValueError:
                sentiment = SentimentLabel.NEUTRAL
                
            # Synthesize polarity/confidence based on mapped enum since LLM doesn't give floats easily here
            polarity_map = {
                SentimentLabel.VERY_POSITIVE: 0.9,
                SentimentLabel.POSITIVE: 0.5,
                SentimentLabel.NEUTRAL: 0.0,
                SentimentLabel.NEGATIVE: -0.5,
                SentimentLabel.VERY_NEGATIVE: -0.9,
            }
            
            return SentimentResult(
                text=text,
                polarity=polarity_map.get(sentiment, 0.0),
                subjectivity=0.8,  # Sarcasm is highly subjective
                sentiment=sentiment,
                confidence=0.95,
                model_used="llm_sarcasm",
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            raise RuntimeError(f"LLM API Error: {str(e)}")
