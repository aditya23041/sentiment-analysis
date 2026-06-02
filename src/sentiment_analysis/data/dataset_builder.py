import logging
from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# GoEmotions has 28 labels. We define them consistently.
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

class MultiTaskDataset(Dataset):
    """
    Unified PyTorch Dataset for multi-task learning (Sarcasm + GoEmotions).
    
    Each item returns:
    - input_ids
    - attention_mask
    - sarcasm_label (0 or 1, or -1 if unknown)
    - emotion_labels (multi-hot encoding of 28 classes, or all -1 if unknown)
    """
    
    def __init__(self, texts: List[str], sarcasm_labels: List[int], emotion_labels: List[List[int]], tokenizer: Any, max_length: int = 128):
        self.texts = texts
        self.sarcasm_labels = sarcasm_labels
        self.emotion_labels = emotion_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        sarcasm = self.sarcasm_labels[idx]
        emotions = self.emotion_labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "sarcasm_label": torch.tensor(sarcasm, dtype=torch.float),
            "emotion_labels": torch.tensor(emotions, dtype=torch.float)
        }

def load_unified_datasets() -> tuple[List[str], List[int], List[List[int]]]:
    """
    Downloads and merges nikesh66/Sarcasm-dataset and google-research-datasets/go_emotions.
    Returns: texts, sarcasm_labels, emotion_labels
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install the datasets library: pip install datasets")
        raise
        
    logger.info("Loading nikesh66/Sarcasm-dataset...")
    # The sarcasm dataset usually has 'text' and 'is_sarcastic' columns
    sarcasm_ds = load_dataset("nikesh66/Sarcasm-dataset", split="train")
    
    logger.info("Loading go_emotions...")
    # GoEmotions has 'text' and 'labels' (a list of integers matching GOEMOTIONS_LABELS)
    goemotions_ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    
    texts = []
    sarcasm_labels = []
    emotion_labels = []
    
    # Process Sarcasm dataset (Emotion labels become -1 mask)
    logger.info("Processing Sarcasm dataset...")
    for item in sarcasm_ds:
        texts.append(item['Tweet'])
        sarcasm_labels.append(1 if item['Sarcasm (yes/no)'] == 'yes' else 0)
        # -1 indicates "ignore this task during loss computation"
        emotion_labels.append([-1] * len(GOEMOTIONS_LABELS))
        
    # Process GoEmotions dataset (Sarcasm label becomes -1 mask)
    logger.info("Processing GoEmotions dataset...")
    for item in goemotions_ds:
        texts.append(item['text'])
        sarcasm_labels.append(-1)
        
        # Create multi-hot encoding for emotions
        multi_hot = [0] * len(GOEMOTIONS_LABELS)
        for label_idx in item['labels']:
            multi_hot[label_idx] = 1
        emotion_labels.append(multi_hot)
        
    logger.info(f"Successfully fused datasets! Total unified records: {len(texts)}")
    return texts, sarcasm_labels, emotion_labels

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_unified_datasets()
