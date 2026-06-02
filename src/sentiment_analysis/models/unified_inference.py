import logging
import os
from typing import Any

from sentiment_analysis.core.memory_graph import memory_graph
from sentiment_analysis.core.schemas import SentimentLabel, SentimentResult
from sentiment_analysis.models.base import SentimentModel

# GoEmotions has 28 labels. We define them consistently here to avoid importing dataset_builder and triggering PyTorch loading.
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

logger = logging.getLogger(__name__)

class UnifiedInferenceModel(SentimentModel):
    """
    Inference wrapper for the custom trained Unified Emotion & Sarcasm Model.
    Utilizes NetworkX memory graph for conversational context.
    """
    def __init__(self, weights_dir: str = "src/sentiment_analysis/data/weights"):
        # LAZY LOAD HEAVY ML LIBRARIES TO PREVENT RENDER OOM ON STARTUP
        global torch, AutoTokenizer, UnifiedEmotionSarcasmModel
        import torch
        from transformers import AutoTokenizer

        from sentiment_analysis.models.unified_trainer import UnifiedEmotionSarcasmModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir

        logger.info(f"Loading unified model on {self.device}...")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)

        # Load Model
        quantized_path = os.path.join(weights_dir, "unified_model_full.pth")
        model_path = os.path.join(weights_dir, "unified_model.pth")

        # 1. Check if quantized model needs to be rebuilt from chunks (or if it's corrupted/empty)
        if not os.path.exists(quantized_path) or os.path.getsize(quantized_path) < 1000000:
            chunk0 = f"{quantized_path}.part0"
            if os.path.exists(chunk0):
                logger.info("Rebuilding quantized model from chunks (streaming 1MB chunks to save memory)...")
                with open(quantized_path, "wb") as outfile:
                    chunk_num = 0
                    while True:
                        chunk_path = f"{quantized_path}.part{chunk_num}"
                        if not os.path.exists(chunk_path):
                            break
                        with open(chunk_path, "rb") as infile:
                            import shutil
                            shutil.copyfileobj(infile, outfile, length=1024*1024)
                        chunk_num += 1
                logger.info("Successfully rebuilt quantized model.")

        if os.path.exists(quantized_path):
            logger.info("Loading FULL INT8 Quantized model directly into RAM (CPU only)...")
            self.device = torch.device("cpu") # INT8 only supports CPU in PyTorch
            self.model = torch.load(quantized_path, map_location="cpu", weights_only=False)
            self.model.to(self.device)
        elif os.path.exists(model_path):
            logger.info(f"Loading original 32-bit model on {self.device}...")
            self.model = UnifiedEmotionSarcasmModel(num_emotion_labels=len(GOEMOTIONS_LABELS), use_config=True)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
        else:
            raise FileNotFoundError("Model weights not found. Please train the model or provide chunks.")

        self.model.eval()

        logger.info("Unified model loaded successfully.")

    @property
    def name(self) -> str:
        return "unified"

    def analyze(self, text: str, **kwargs: Any) -> SentimentResult:
        session_id = kwargs.get("session_id", "default_session")

        # 1. Check Memory Graph Context
        history = memory_graph.get_session_history(session_id, limit=3)
        " ".join([item["text"] for item in history])

        # We can append context to the text for the transformer,
        # but for simplicity and safety, we just analyze the current text
        # and adjust the sarcasm threshold if the user was previously highly emotional.
        # (This is a simplified use of the memory graph for inference).

        # 2. Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(self.device)

        # 3. Inference
        with torch.no_grad():
            sarcasm_pred, emotion_pred = self.model(inputs["input_ids"], inputs["attention_mask"])

        sarcasm_score = sarcasm_pred.item()
        emotions = emotion_pred.squeeze().cpu().numpy()

        # 4. Process Results
        # Get top 3 emotions
        top_indices = emotions.argsort()[-3:][::-1]
        top_emotions = {GOEMOTIONS_LABELS[i]: float(emotions[i]) for i in top_indices if emotions[i] > 0.3}

        # Adjust sarcasm threshold based on memory context
        # If they were highly emotional recently, they are more likely to be sarcastic now
        sarcasm_threshold = 0.5
        if len(history) > 0 and len(history[-1].get("emotions", {})) > 0:
            sarcasm_threshold = 0.4

        is_sarcastic = sarcasm_score > sarcasm_threshold

        # Determine base polarity from GoEmotions (simplified mapping)
        positive_labels = {"admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
        negative_labels = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}

        polarity = 0.0
        for emotion, score in top_emotions.items():
            if emotion in positive_labels:
                polarity += score
            elif emotion in negative_labels:
                polarity -= score

        # Cap polarity between -1.0 and 1.0
        polarity = max(-1.0, min(1.0, polarity))

        # If sarcastic, invert polarity (e.g. "Great job breaking production" -> positive words, but sarcastic -> negative)
        if is_sarcastic:
            polarity = -polarity

        # Determine label
        if polarity > 0.2:
            label = SentimentLabel.POSITIVE
        elif polarity < -0.2:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        confidence = float(max(abs(polarity), sarcasm_score))

        # 5. Store in Memory Graph
        memory_graph.add_utterance(session_id, text, top_emotions)

        return SentimentResult(
            text=text,
            polarity=polarity,
            sentiment=label,
            confidence=confidence,
            model_used=self.name,
            is_sarcastic=is_sarcastic,
            sarcasm_probability=sarcasm_score,
            metadata={"emotions": top_emotions, "context_aware": len(history) > 0}
        )
