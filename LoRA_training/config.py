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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("GLUE_Training")
if not logger.handlers: 
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler("output/logger_training.log", maxBytes=1024*1024, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


__all__ = ["logger", "DEVICE"]

