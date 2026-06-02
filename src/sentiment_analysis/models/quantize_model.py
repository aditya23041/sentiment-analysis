import os
import sys

# Import datasets FIRST to avoid Windows 11 C++ PyArrow/PyTorch segfault
import datasets

import torch
import logging
from transformers import AutoTokenizer

# Temporarily insert src to sys.path so we can import the model
sys.path.insert(0, os.path.abspath('src'))

from sentiment_analysis.models.unified_trainer import UnifiedEmotionSarcasmModel
from sentiment_analysis.data.dataset_builder import GOEMOTIONS_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize():
    weights_dir = "src/sentiment_analysis/data/weights"
    model_path = os.path.join(weights_dir, "unified_model.pth")
    quantized_path = os.path.join(weights_dir, "unified_model_quantized.pth")
    
    if not os.path.exists(model_path):
        logger.error(f"Original model not found at {model_path}")
        return
        
    logger.info("Loading original 32-bit model into CPU memory...")
    # Quantization must be done on CPU
    device = torch.device("cpu")
    model = UnifiedEmotionSarcasmModel(num_emotion_labels=len(GOEMOTIONS_LABELS))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    logger.info("Applying INT8 Dynamic Quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    logger.info("Saving quantized model weights...")
    torch.save(quantized_model.state_dict(), quantized_path)
    
    # Compare sizes
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
    
    logger.info(f"Original Size: {orig_size:.2f} MB")
    logger.info(f"Quantized Size: {quant_size:.2f} MB")
    logger.info("Quantization complete! Generating split chunks for GitHub compatibility...")
    
    # Split the quantized file into chunks to bypass GitHub's 100MB limit
    CHUNK_SIZE = 50 * 1024 * 1024 # 50MB
    with open(quantized_path, "rb") as f:
        chunk_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            chunk_path = f"{quantized_path}.part{chunk_num}"
            with open(chunk_path, "wb") as chunk_file:
                chunk_file.write(chunk)
            logger.info(f"Created chunk: {chunk_path}")
            chunk_num += 1
            
    # Delete the large unified_model_quantized.pth so it doesn't get pushed
    os.remove(quantized_path)
    logger.info("Deleted unified_model_quantized.pth to prevent accidental git push. Chunks are ready!")

if __name__ == "__main__":
    quantize()
