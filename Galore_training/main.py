import torch
from config import *
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_preprocess_data
from train import setup_trainer

def main():
    try:
        logger.info(f"Hyperparameters - Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, "
                    f"Epochs: {EPOCHS}, Seed: {SEED}, "
                    f"Galore Rank: {RANK}, Update Projection Gap: {UPDATE_PROJ_GAP}, "
                    f"Scale: {SCALE}, Projection Type: {PROJ_TYPE}, Optimizer: {OPTIM}, "
                    f"Optimizer Target Modules: {OPTIM_TARGET_MODULES}, "
                    f"Model: {MODEL_NAME}, Dataset: {DATASET_NAME}, Device: {DEVICE}")

        logger.info(f"Initializing model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
        # Load and preprocess dataset
        logger.info(f"Retrieving dataset: {DATASET_NAME}")
        train_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME, MODEL_NAME)
        
        # Setting up trainer
        logger.info("Setting up trainer")
        trainer = setup_trainer(model, train_dataset=train_dataset, test_dataset=test_dataset)
        
        logger.info("Starting training")
        trainer.train()

        model.eval()  
        torch.save(model.state_dict(), "output/fine_tuned_bert_hpa.pt")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise 
        

if __name__ == "__main__":
    main()