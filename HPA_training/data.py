# Data loading & preprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from config import *
from torch.utils.data import DataLoader

def load_and_preprocess_data(dataset_name, model_name):
    logger.info(f"Loading and preprocessing {dataset_name} dataset")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    encoded_dataset = load_dataset(dataset_name).map(preprocess_function, batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    train_dataset = encoded_dataset["train"]
    test_dataset = encoded_dataset["test"]
    logger.info("Dataset loaded and preprocessed successfully")
    return train_dataset, test_dataset
