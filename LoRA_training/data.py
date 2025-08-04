# Data loading & preprocessing
import argparse
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config import logger, Config_Args
from helper import TASK_TO_COLUMNS, TASK_TO_LABELS
from torch.utils.data import DataLoader

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
    NUM_LABELS = TASK_TO_LABELS[args.task_name]
    def preprocess_fn(examples):
        input_columns = TASK_TO_COLUMNS[args.task_name]
        print(f"input_columns: {input_columns}")
        if len(input_columns) == 1:
            # Single-sentence task
            preprocessed = tokenizer(
                examples[input_columns[0]],
                # padding=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=args.max_seq_length
                # max_length=tokenizer.model_max_length
            )
        else:
            # Sentence-pair task
            preprocessed = tokenizer(
                examples[input_columns[0]],
                examples[input_columns[1]],
                # padding=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=args.max_seq_length
                # max_length=tokenizer.model_max_length
            )

        return preprocessed
    
    # Tokenize dataset
    logger.info(f"Loading and preprocessing {args.dataset_name} dataset")
    if not (args.task_name == None):
        if args.task_name == "mnli":
            splits = ["train", "validation_matched", "validation_mismatched"]
            dataset = load_dataset(args.dataset_name, args.task_name, split=splits)
        else:
            splits = ["train","validation"]
            dataset = load_dataset(args.dataset_name, args.task_name, split=splits)
    else:
        dataset = load_dataset(args.dataset_name)

    dataset = DatasetDict({split: ds for split, ds in zip(splits, dataset)})
        

    encoded_dataset = dataset.map(preprocess_fn, batched=True)
    
    # Format for PyTorch and rename label column
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    
    # Get train and validation splits
    train_dataset = encoded_dataset["train"]
    # train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)

    if args.task_name == "mnli":
        validation_matched_dataset = encoded_dataset["validation_matched"]
        validation_mismatched_dataset = encoded_dataset["validation_mismatched"]

        
        # validation_matched_dataset = DataLoader(validation_matched_dataset, batch_size=32, shuffle=True)
        # validation_mismatched_dataset = DataLoader(validation_mismatched_dataset, batch_size=32, shuffle=True)
    else:
        test_dataset = encoded_dataset["validation"] # Since glue test dataset does not have labels
        # test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    task_name_string = "None" if args.task_name == None else args.task_name

    logger.info(f"{args.dataset_name} Dataset for {task_name_string} loaded and preprocessed successfully")

    if args.task_name == "mnli":
        return train_dataset, validation_matched_dataset, validation_mismatched_dataset, tokenizer
    else:
        return train_dataset, test_dataset, tokenizer
