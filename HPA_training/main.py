import torch
from config import *
from data import load_and_preprocess_data
from transformers import AutoModelForSequenceClassification
from train import HpaTrainer
from hpa import replace_with_hpa, make_flexi_optimizer

      
def main():
    try:
        logger.info(f"Initializing model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        logger.info(f"Retrieving dataset: {DATASET_NAME}")
        train_dataset, test_dataset = load_and_preprocess_data(DATASET_NAME, MODEL_NAME)

        logger.info("Replacing target modules with HPA")
        replace_with_hpa(model, set_rank_fn=lambda m,n: INITIAL_RANK, hpa_dropout=LORA_DROPOUT, hpa_alpha=LORA_ALPHA)
        
        model.to(DEVICE)

        logger.info("Initializing optimizer")
        FS_AdamW = make_flexi_optimizer(torch.optim.AdamW, ["exp_avg", "exp_avg_sq"])
        optimizer = FS_AdamW(
            model,
            model.named_parameters(), 
            lr=LEARNING_RATE
        )
        
        # Train the model
        trainer = HpaTrainer(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            warmup_steps = 0,
            gradient_accumulation_steps = 1, 
            refresh_adapter_steps = None, 
            refresh_type = "weight",
            rank_reduction_cycles = None,
            first_rank_reduction_cycle = None,
            output_dir = './results',
            logging_dir = './logs',
            logging_steps = 10,
            device = None, 
            lr_scheduler = None, 
            max_grad_norm = 1.0 
        )

        trainer.train()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise 
        

if __name__ == "__main__":
    main()