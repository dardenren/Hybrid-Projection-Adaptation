import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from config import Config_Args, DEVICE
from metrics import SystemMetricsCallback

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert NumPy arrays to PyTorch Tensors
    predictions = torch.tensor(predictions, device=DEVICE)
    labels = torch.tensor(labels, device=DEVICE)
    # Compute loss
    loss = nn.CrossEntropyLoss()(predictions, labels)
    return {"eval_loss": loss.item()}

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],  # Only optimize LoRA parameters
        lr=Config_Args.args.lr,      
        # betas=(0.9, 0.999),    
        # eps=1e-8,             
        # weight_decay=0.01,     
        # amsgrad=False         
    )
    output_dir_string = "./output" + f"_{Config_Args.args.task_name}"
    logging_dir_string = "./logs" + f"_{Config_Args.args.task_name}"

    training_args = TrainingArguments(
        # output_dir="./output",
        output_dir=output_dir_string,
        num_train_epochs=Config_Args.args.epochs,
        per_device_train_batch_size=Config_Args.args.train_batch_size,
        learning_rate=Config_Args.args.lr,
        warmup_ratio = 0.1,
        lr_scheduler_type="linear",
        seed=Config_Args.args.seed,
        # logging_dir="./logs",
        logging_dir=logging_dir_string,
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