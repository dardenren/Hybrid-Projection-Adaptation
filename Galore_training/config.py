import torch
import logging
from logging.handlers import RotatingFileHandler
import argparse

# ArgumentParser object
class Config_Args:
    args = None

    @classmethod
    def update_args(cls, args_obj):
        cls.args = args_obj

    @classmethod
    def get_args(cls):
        return cls.args



args = None

# Hyperparameters
SEED = 888

#Galore parameters
RANK = 8  
UPDATE_PROJ_GAP = 200
SCALE = 0.25
PROJ_TYPE = "std"
OPTIM = "galore_adamw"
OPTIM_TARGET_MODULES = ["attention", "pooler", "intermediate"] 
# ["attn", "mlp"]

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
    "DEVICE",
    "OPTIM",
    "OPTIM_TARGET_MODULES",
    "PROJ_TYPE",
    "RANK",
    "SCALE",
    "SEED",
    "UPDATE_PROJ_GAP",
    "logger"
]
