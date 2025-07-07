# Data loading & preprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from config import logger, BATCH_SIZE, MODEL_NAME, DATASET_NAME
from torch.utils.data import DataLoader

def load_and_preprocess_data(dataset_name=DATASET_NAME, model_name=MODEL_NAME):
    logger.info(f"Loading and preprocessing {dataset_name} dataset")
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    train_dataset = encoded_dataset["train"]
    test_dataset = encoded_dataset["test"]
    logger.info("Dataset loaded and preprocessed successfully")
    train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataset, test_dataset
