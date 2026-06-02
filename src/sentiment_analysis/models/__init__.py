"""Sentiment analysis model backends."""

from sentiment_analysis.models.base import SentimentModel
from sentiment_analysis.models.llm_model import LLMSentimentModel
from sentiment_analysis.models.textblob_model import TextBlobModel
from sentiment_analysis.models.transformer_model import TransformerModel
from sentiment_analysis.models.vader_model import VaderModel

# Model registry — maps string names to model classes
MODEL_REGISTRY: dict[str, type[SentimentModel]] = {
    "textblob": TextBlobModel,
    "vader": VaderModel,
    "transformer": TransformerModel,
    "llm": LLMSentimentModel,
}



def get_model(name: str, **kwargs: object) -> SentimentModel:
    """Get a sentiment model by name.

    Args:
        name: Model name ('textblob', 'vader', 'transformer')
        **kwargs: Additional keyword arguments passed to the model constructor

    Returns:
        Initialized model instance

    Raises:
        ValueError: If the model name is not recognized
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List available model names."""
    return sorted(MODEL_REGISTRY.keys())


__all__ = [
    "MODEL_REGISTRY",
    "SentimentModel",
    "TextBlobModel",
    "VaderModel",
    "get_model",
    "list_models",
]
