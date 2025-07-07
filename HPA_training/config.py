import torch
import logging
from logging.handlers import RotatingFileHandler

# Hyperparameters
LEARNING_RATE = 1e-4
INITIAL_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0

BATCH_SIZE = 2
EPOCHS = 2

MODEL_NAME = "google/mobilebert-uncased"
DATASET_NAME = "imdb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("MobileBERT_Training")
if not logger.handlers:  # Avoid reconfiguring if already set
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler("output/training.log", maxBytes=1024*1024, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


__all__ = ["logger", "LEARNING_RATE", "INITIAL_RANK", "LORA_ALPHA", "LORA_DROPOUT", "BATCH_SIZE", "EPOCHS", "MODEL_NAME", "DATASET_NAME", "DEVICE"]

