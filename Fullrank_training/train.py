import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED
from metrics import SystemMetricsCallback

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,      
        betas=(0.9, 0.98),    
        eps=1e-7,             
        weight_decay=0.1,     
        amsgrad=False         
    )

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        logging_dir="./logs",
        report_to="tensorboard",
        logging_steps=100,  # Log every 100 steps
        logging_strategy="steps",
        save_strategy="no", # Set to "epoch" to save model after every epoch
        eval_strategy="steps",
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=nn.CrossEntropyLoss(),
        optimizers=(optimizer, None),
        callbacks=[SystemMetricsCallback(log_dir=training_args.logging_dir, model=model, tokenizer=tokenizer)]
    )

    return trainer