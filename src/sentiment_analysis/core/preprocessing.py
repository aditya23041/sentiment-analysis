"""Text preprocessing pipeline for sentiment analysis."""

from __future__ import annotations

import html
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK data is available
for _resource, _name in [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_name, quiet=True)


# Common English contractions
_CONTRACTIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "you're": "you are",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "let's": "let us",
    "'ve": " have",
    "'re": " are",
    "'ll": " will",
    "'d": " would",
}

# Compiled regex patterns
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_EXTRA_WHITESPACE = re.compile(r"\s+")
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")


class TextPreprocessor:
    """Configurable text preprocessing pipeline.

    Applies a sequence of cleaning steps to normalize input text
    before sentiment analysis. Steps can be toggled on or off.

    Args:
        remove_urls: Strip HTTP/HTTPS URLs. Default True.
        remove_html: Strip HTML tags and decode entities. Default True.
        remove_emails: Strip email addresses. Default True.
        remove_mentions: Strip @mentions. Default True.
        expand_hashtags: Convert #CamelCase to separate words. Default True.
        expand_contractions: Expand can't -> cannot, etc. Default True.
        remove_stopwords: Remove common English stopwords. Default False.
        normalize_repeated: Reduce repeated chars (e.g. "soooo" -> "soo"). Default True.
        lowercase: Convert to lowercase. Default True.
    """

    def __init__(
        self,
        *,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        expand_hashtags: bool = True,
        expand_contractions: bool = True,
        remove_stopwords: bool = False,
        normalize_repeated: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.expand_hashtags = expand_hashtags
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.normalize_repeated = normalize_repeated
        self.lowercase = lowercase

        self._stop_words: set[str] | None = None

    @property
    def stop_words(self) -> set[str]:
        """Lazily load stopwords."""
        if self._stop_words is None:
            self._stop_words = set(stopwords.words("english"))
        return self._stop_words

    def process(self, text: str) -> str:
        """Apply the full preprocessing pipeline to text.

        Args:
            text: Raw input text.

        Returns:
            Cleaned, normalized text.
        """
        if not text or not text.strip():
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKD", text)

        # Strip HTML
        if self.remove_html:
            text = _HTML_TAG_PATTERN.sub(" ", text)
            text = html.unescape(text)

        # Remove URLs
        if self.remove_urls:
            text = _URL_PATTERN.sub(" ", text)

        # Remove emails
        if self.remove_emails:
            text = _EMAIL_PATTERN.sub(" ", text)

        # Remove @mentions
        if self.remove_mentions:
            text = _MENTION_PATTERN.sub(" ", text)

        # Expand hashtags: #GreatProduct -> Great Product
        if self.expand_hashtags:
            text = _HASHTAG_PATTERN.sub(
                lambda m: " ".join(re.findall(r"[A-Z][a-z]*|[a-z]+|[A-Z]+", m.group(1))),
                text,
            )

        # Expand contractions
        if self.expand_contractions:
            for contraction, expanded in _CONTRACTIONS.items():
                text = text.replace(contraction, expanded)
                text = text.replace(contraction.title(), expanded.title())

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Normalize repeated characters
        if self.normalize_repeated:
            text = _REPEATED_CHARS.sub(r"\1\1", text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
            text = " ".join(tokens)

        # Clean up whitespace
        text = _EXTRA_WHITESPACE.sub(" ", text).strip()

        return text
