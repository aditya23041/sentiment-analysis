"""Sentiment analysis model backends."""

import logging

from sentiment_analysis.models.base import SentimentModel
from sentiment_analysis.models.llm_model import LLMSentimentModel
from sentiment_analysis.models.textblob_model import TextBlobModel
from sentiment_analysis.models.transformer_model import TransformerModel
from sentiment_analysis.models.vader_model import VaderModel

logger = logging.getLogger(__name__)

# Model registry — maps string names to model classes
# NOTE: 'unified' is NOT registered here because importing it triggers PyTorch,
# which uses >400MB RAM and crashes Render's 512MB free tier.
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
    # Lazy-load the unified model only when explicitly requested
    if name == "unified" and name not in MODEL_REGISTRY:
        try:
            from sentiment_analysis.models.unified_inference import UnifiedInferenceModel
            MODEL_REGISTRY["unified"] = UnifiedInferenceModel
            logger.info("Unified model registered (PyTorch loaded on demand)")
        except ImportError:
            raise ValueError(
                "Unified model requires PyTorch. Install with: pip install torch"
            ) from None

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
