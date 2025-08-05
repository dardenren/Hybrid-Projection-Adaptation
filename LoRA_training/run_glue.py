import torch
import argparse
from config import Config_Args
from config import *
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_preprocess_data
from train import setup_trainer
from helper import TASK_TO_LABELS, TASK_TO_COLUMNS
from eval import evaluate_model 
from model import load_model, merge_lora_weights
from lora import mark_only_lora_as_trainable, lora_state_dict

def main(args):
    try:
        NUM_LABELS = TASK_TO_LABELS[args.task_name]
        logger.info(f"Hyperparameters - LoRA Rank: {args.rank}, LoRA Alpha: {args.scale}, "
                    f"Merge Weights: {args.merge_weights}, Learning Rate: {args.lr}, "
                    f"Batch Size: {args.train_batch_size}, Epochs: {args.epochs}, "
                    f"Model: {args.model_name}, Dataset: {args.dataset_name}, "
                    f"SEED Size: {args.seed}, Device: {DEVICE}")

        logger.info(f"Initializing model: {args.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, 
                                                            num_labels=NUM_LABELS)
        
        # model = load_model(model_name=args.model_name, r=args.rank, lora_alpha=args.scale, 
        #                 merge_weights=False, linear=True, embedding=True)
        
        # # Mark only LoRA parameters as trainable
        # mark_only_lora_as_trainable(model)

        model.to(DEVICE)

    
        # Load and preprocess dataset
        task_name_string = "None" if args.task_name == None else args.task_name
        logger.info(f"Retrieving dataset: {args.dataset_name}, Task: {task_name_string}")
        if args.task_name == "mnli":
            # test_dataset is the validation_matched_dataset in mnli
            train_dataset, test_dataset, validation_mismatched_dataset, tokenizer = load_and_preprocess_data(args)
        else:
            train_dataset, test_dataset, tokenizer = load_and_preprocess_data(args)
        
        # Setting up trainer
        logger.info("Setting up trainer")
        trainer = setup_trainer(model, tokenizer, train_dataset=train_dataset, test_dataset=test_dataset)

        logger.info("Starting training")
        trainer.train()

        model_name_replaced = args.model_name.replace("/", "-")
        save_path = "output/fine-tuned" + f"_{model_name_replaced}" + "_full-rank.pt" 
        lora_params = lora_state_dict(model)
        torch.save(lora_params, f"output/lora_params_{model_name_replaced}.pt")

        # if args.merge_weights:
        #     logger.info("Merging LoRA weights")
        #     merge_lora_weights(model)
        model = trainer.model.merge_and_unload()
        model.eval()
        torch.save(model.state_dict(), save_path)

        if args.task_name == "mnli":
            print("Matched results: \n")
            evaluate_model(model, test_dataset)

            print("Mismatched results: \n")
            evaluate_model(model, validation_mismatched_dataset)
        else:
            evaluate_model(model, test_dataset)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="Dataset name (e.g., nyu-mll/glue)")
    parser.add_argument("--task_name", required=True, help="GLUE task name (e.g., cola, mnli)")
    parser.add_argument("--model_name", required=True, help="Model name (e.g., bert-base-uncased)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Seed for randomization")
    parser.add_argument("--rank", type=int, default=8, help="Rank for low rank gradient/adapter")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for low rank gradient/adapter, for LoRA should be approximately double the rank")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Number of tokens converted into embedding")
    parser.add_argument("--merge_weights", type=bool, default=True, help="Option to merge LoRA matrices with weight matrices during inference")
    args = parser.parse_args()
    Config_Args.update_args(args)
    main(args)
