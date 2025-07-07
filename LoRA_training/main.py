import os
if (not os.path.isdir("output")):
    os.mkdir("output")


import torch
import logging
from config import *
from data import load_and_preprocess_data
from model import load_model
from lora import mark_only_lora_as_trainable, lora_state_dict
from train import train_model


      
def main():
    try:
        # Load data
        train_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME, MODEL_NAME)
        
        # Load and adapt model
        model = load_model(model_name=MODEL_NAME, type="lora", r=LORA_RANK, 
                        lora_alpha=LORA_ALPHA, merge_weights=MERGE_WEIGHTS, linear= True, embedding=False)
        
        # Mark only LoRA parameters as trainable
        mark_only_lora_as_trainable(model)
        
        # Move model to device
        model.to(DEVICE)

        # Optimizer for model
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],  # Only optimize LoRA parameters
            lr=LEARNING_RATE
        )
        
        # Train the model
        train_model(model, train_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, optimizer, DEVICE)

        # Save LoRA parameters
        lora_params = lora_state_dict(model)
        torch.save(lora_params, "output/lora_params_imdb.pt")
        
        # Save the model
        model.eval()  # Merge LoRA weights
        torch.save(model.state_dict(), "output/fine_tuned_bert_lora.pt")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise 
        

if __name__ == "__main__":
    main()