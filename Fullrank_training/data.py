# Data loading & preprocessing
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from config import logger, Config_Args
from helper import TASK_TO_COLUMNS, TASK_TO_LABELS
# from torch.utils.data import DataLoader

# def load_and_preprocess_data(dataset_name="nyu-mll/glue", model_name="google/mobilebert-uncased", columns=):
#     logger.info(f"Loading and preprocessing {dataset_name} dataset")
#     dataset = load_dataset(dataset_name)
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)

#     def preprocess_function(examples):
#         return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

#     encoded_dataset = dataset.map(preprocess_function, batched=True)
#     encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
#     encoded_dataset = encoded_dataset.rename_column("label", "labels")
#     train_dataset = encoded_dataset["train"]
#     test_dataset = encoded_dataset["test"]
#     logger.info("Dataset loaded and preprocessed successfully")
#     return train_dataset, test_dataset

def load_and_preprocess_data(args: argparse.ArgumentParser):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess_fn(examples):
        input_columns = TASK_TO_COLUMNS[args.task_name]
        if len(input_columns) == 1:
            # Single-sentence task
            preprocessed_tokenizer = tokenizer(
                examples[input_columns[0]],
                # padding=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=512
                # max_length=tokenizer.model_max_length
            )
        else:
            # Sentence-pair task
            preprocessed_tokenizer = tokenizer(
                examples[input_columns[0]],
                examples[input_columns[1]],
                # padding=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=512
                # max_length=tokenizer.model_max_length
            )

        logger.debug(f"Sample input_ids length: {len(preprocessed_tokenizer['input_ids'][0])}")
        return preprocessed_tokenizer
    
    # Tokenize dataset
    logger.info(f"Loading and preprocessing {args.dataset_name} dataset")
    if not (args.task_name == None):
        dataset = load_dataset(args.dataset_name, args.task_name)
    else:
        dataset = load_dataset(args.dataset_name)
        

    encoded_dataset = dataset.map(preprocess_fn, batched=True)
    
    # Format for PyTorch and rename label column
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    
    # Get train and validation splits
    train_dataset = encoded_dataset["train"]
    # validation_dataset = encoded_dataset["validation"]
    test_dataset = encoded_dataset["test"]
    
    task_name_string = "None" if args.task_name == None else args.task_name
    for split, ds in [("train", train_dataset), ("test", test_dataset)]:
            lengths = [len(item["input_ids"]) for item in ds]
            if len(set(lengths)) > 1:
                logger.warning(f"Inconsistent sequence lengths in {split} split: {set(lengths)}")

    logger.info(f"{args.dataset_name} Dataset for {task_name_string} loaded and preprocessed successfully")
    # return train_dataset, validation_dataset, test_dataset, tokenizer
    return train_dataset, test_dataset, tokenizer
