import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from config import Config_Args
from metrics import SystemMetricsCallback

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert NumPy arrays to PyTorch Tensors
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    # Compute loss
    loss = nn.CrossEntropyLoss()(predictions, labels)
    return {"eval_loss": loss.item()}

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config_Args.args.lr,      
        # betas=(0.9, 0.999),    
        # eps=1e-8,             
        # weight_decay=0.01,     
        # amsgrad=False         
    )

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=Config_Args.args.epochs,
        per_device_train_batch_size=Config_Args.args.train_batch_size,
        learning_rate=Config_Args.args.lr,
        seed=Config_Args.args.seed,
        logging_dir="./logs",
        report_to="tensorboard",
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
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[SystemMetricsCallback(log_dir=training_args.logging_dir, model=model, tokenizer=tokenizer)]
    )

    return trainer