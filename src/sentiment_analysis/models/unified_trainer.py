import os
import sys
sys.path.insert(0, os.path.abspath('src'))

import logging

# IMPORTANT: On Windows 11, PyArrow (used by datasets) MUST be imported before PyTorch
# to prevent a C++ DLL memory access violation (segfault) during initialization.
import datasets

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

# Default hyperparams optimized for RTX 4050 (6GB VRAM)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MODEL_NAME = "distilbert-base-uncased" # Lightweight base model

class UnifiedEmotionSarcasmModel(nn.Module):
    def __init__(self, num_emotion_labels=28, use_config=False):
        super().__init__()
        if use_config:
            # Initialize random weights from config (saves 260MB RAM & download time on Render)
            config = AutoConfig.from_pretrained(MODEL_NAME)
            self.transformer = AutoModel.from_config(config)
        else:
            self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        
        # Sarcasm is binary (0 or 1) -> 1 output node with Sigmoid
        self.sarcasm_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.transformer.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Emotions are multi-label -> 28 output nodes with Sigmoid
        self.emotion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.transformer.config.hidden_size, num_emotion_labels),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        sarcasm_pred = self.sarcasm_head(cls_output)
        emotion_pred = self.emotion_head(cls_output)
        
        return sarcasm_pred, emotion_pred

def train_model():
    """
    Main training loop for the unified model. 
    Designed to be run manually by the user on their RTX 4050.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Custom Unified Training (CUDA optimized)...")
    
    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    if device.type != "cuda":
        logger.warning("CUDA is not available. Training on CPU will be extremely slow. "
                       "Ensure PyTorch with CUDA is installed (pip install torch --index-url https://download.pytorch.org/whl/cu121).")

    # 2. Load Datasets
    logger.info("Loading unified datasets from HuggingFace...")
    print("DEBUG: Before importing dataset_builder")
    from sentiment_analysis.data.dataset_builder import load_unified_datasets, MultiTaskDataset
    print("DEBUG: After importing dataset_builder")
    texts, sarcasm_labels, emotion_labels = load_unified_datasets()
    print("DEBUG: After load_unified_datasets")
    
    # 3. Tokenization & DataLoader
    logger.info("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = MultiTaskDataset(texts, sarcasm_labels, emotion_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Model Setup
    model = UnifiedEmotionSarcasmModel().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Loss Functions
    # Use BCEWithLogitsLoss if we drop Sigmoid, but since we have Sigmoid, we use BCELoss
    # IMPORTANT: We use reduction='none' to mask out the -1 labels
    loss_fn = nn.BCELoss(reduction='none')
    
    logger.info(f"Starting training for {EPOCHS} epochs...")
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_sarcasm = batch['sarcasm_label'].to(device)
            target_emotions = batch['emotion_labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_sarcasm, pred_emotions = model(input_ids, attention_mask)
            
            # Compute Sarcasm Loss (Ignore -1 labels)
            sarcasm_mask = (target_sarcasm != -1)
            loss_sarcasm = 0
            if sarcasm_mask.any():
                s_pred = pred_sarcasm.squeeze()[sarcasm_mask]
                s_target = target_sarcasm[sarcasm_mask]
                loss_sarcasm = loss_fn(s_pred, s_target).mean()
                
            # Compute Emotion Loss (Ignore -1 labels)
            # Find rows where emotion_labels doesn't have -1
            emotion_mask = (target_emotions[:, 0] != -1)
            loss_emotion = 0
            if emotion_mask.any():
                e_pred = pred_emotions[emotion_mask]
                e_target = target_emotions[emotion_mask]
                loss_emotion = loss_fn(e_pred, e_target).mean()
                
            # Combined Loss (equal weighting)
            loss = loss_sarcasm + loss_emotion
            
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if step % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(dataloader)} | Loss: {loss:.4f}")
                
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(dataloader):.4f}")
        
    # Save Model Weights
    output_dir = "src/sentiment_analysis/data/weights"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "unified_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save the tokenizer so it matches the model
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training Complete! Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()
