import torch
import logging
from logging.handlers import RotatingFileHandler

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 4

SEED = 888
NUM_LABELS = 2  # Adjust based on task

#Galore parameters
RANK = 8  
UPDATE_PROJ_GAP = 200
SCALE = 0.25
PROJ_TYPE = "std"
OPTIM = "galore_adamw"
OPTIM_TARGET_MODULES = ["attention", "pooler", "intermediate"] 
# ["attn", "mlp"]

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


__all__ = [
    "BATCH_SIZE",
    "DATASET_NAME",
    "DEVICE",
    "EPOCHS",
    "LEARNING_RATE",
    "MODEL_NAME",
    "NUM_LABELS",
    "OPTIM",
    "OPTIM_TARGET_MODULES",
    "PROJ_TYPE",
    "RANK",
    "SCALE",
    "SEED",
    "UPDATE_PROJ_GAP",
    "logger"
]
