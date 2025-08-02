import torch
import argparse
from config import Config_Args
from config import *
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_preprocess_data
from train import setup_trainer
from helper import TASK_TO_LABELS, TASK_TO_COLUMNS
from eval import evaluate_model

def main(args):
    try:
        NUM_LABELS = TASK_TO_LABELS[args.task_name]
        logger.info(f"Hyperparameters - Learning Rate: {args.lr}, Batch Size: {args.train_batch_size}, "
                    f"Epochs: {args.epochs}, Seed: {args.seed}, Num Labels: {NUM_LABELS}, "
                    f"Model: {args.model_name}, Dataset/Task: {args.task_name}, Device: {DEVICE}")

        logger.info(f"Initializing model: {args.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, 
                                                                   num_labels=NUM_LABELS)
    
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
        trainer = setup_trainer(model, tokenizer, train_dataset, test_dataset)
        
        logger.info("Starting training")
        trainer.train()

        model.eval()  
        
        model_name_replaced = args.model_name.replace("/", "-")
        save_path = "output/fine-tuned" + f"_{model_name_replaced}" + "_full-rank.pt" 
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Seed for randomization")
    args = parser.parse_args()
    Config_Args.update_args(args)
    main(args)
