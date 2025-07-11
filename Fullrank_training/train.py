import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED

def setup_trainer(model, train_dataset, test_dataset):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,      
        betas=(0.9, 0.98),    
        eps=1e-7,             
        weight_decay=0.1,     
        amsgrad=False         
    )

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        logging_dir="output/training.log",
        logging_steps=100,  # Log every 100 steps
        logging_strategy="steps",
        save_strategy="no", # Set to "epoch" to save model after every epoch
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=nn.CrossEntropyLoss(),
        optimizers=(optimizer, None)
    )

    return trainer