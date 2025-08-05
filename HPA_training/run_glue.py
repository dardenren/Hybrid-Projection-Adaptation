import torch
import argparse
from config import Config_Args
from config import *
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_preprocess_data
from helper import TASK_TO_LABELS, TASK_TO_COLUMNS
from eval import evaluate_model 
from trainer import HpaTrainer
from hpa import replace_with_hpa, make_flexi_optimizer
from train import setup_trainer


def main(args):
    try:
        NUM_LABELS = TASK_TO_LABELS[args.task_name]
        logger.info(f"Hyperparameters - Learning Rate: {args.lr}, Batch Size: {args.train_batch_size}, "
                    f"Epochs: {args.epochs}, Seed: {args.seed}, HPA_MODE: {args.hpa_mode}, "
                    f"HPA Rank: {args.rank}, HPA Dropout: {args.hpa_dropout}, Update Projection Gap: {args.proj_freq}, "
                    f"Scale: {args.scale}, Projection Type: {args.proj_type}, Specified Projection Direction: {args.proj_direction}, "
                    f"Optimizer Target Modules: {OPTIM_TARGET_MODULES}, "
                    f"Model: {args.model_name}, Dataset: {args.dataset_name}, Device: {DEVICE}")

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
        
        # Add HPA adapters
        logger.info("Adding HPA to target modules")
        if args.proj_direction == None:
            replace_with_hpa(model, OPTIM_TARGET_MODULES, set_rank_fn=lambda m,n: args.rank, mode="svd", hpa_dropout=args.hpa_dropout, hpa_alpha=args.scale)
        else:
            replace_with_hpa(model, OPTIM_TARGET_MODULES, set_rank_fn=lambda m,n: args.rank, mode="svd", hpa_dropout=args.hpa_dropout, hpa_alpha=args.scale, proj_direction=args.proj_direction)

        model.to(DEVICE)

        # Setting up trainer
        logger.info("Setting up trainer")
        trainer = setup_trainer(model, tokenizer, train_dataset, test_dataset)
        
        logger.info("Starting training")
        trainer.train()

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
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Seed for randomization")
    parser.add_argument("--rank", type=int, default=8, help="Rank for low rank gradient/adapter")
    parser.add_argument("--proj_freq", type=int, default=None, help="Steps per update of projection matrix")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for low rank gradient/adapter e.g. LoRA alpha")
    parser.add_argument("--proj_type", type=str, default="std", help="Method to obtain projection vectors")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Number of tokens converted into embedding")
    parser.add_argument("--hpa_mode", type=str, default="svd", help="Mode for HPA, one of ['svd', 'random', 'random_sample', 'lora', 'random_svd_(int)', 'power_sample_(int)']")
    parser.add_argument("--proj_direction", type=str, default=None, help="Projection direction for HPA adapters, one of ['left', 'right', 'thin_left', 'thin_right', 'fat_right', 'fat_left', 'both', 'diag']")
    parser.add_argument("--hpa_dropout", type=float, default=0, help="Dropout value for HPA adapters")
    args = parser.parse_args()
    Config_Args.update_args(args)
    main(args)
