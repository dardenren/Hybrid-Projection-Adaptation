import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from config import Config_Args
from config import OPTIM, OPTIM_TARGET_MODULES, PROJ_TYPE, RANK, SCALE, SEED, UPDATE_PROJ_GAP
from metrics import SystemMetricsCallback

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert NumPy arrays to PyTorch Tensors
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    loss = nn.CrossEntropyLoss()(predictions, labels)
    return {"eval_loss": loss.item()}

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=Config_Args.args.epochs,
        per_device_train_batch_size=Config_Args.args.train_batch_size,
        learning_rate=Config_Args.args.lr,
        seed=SEED,
        optim=OPTIM,
        optim_args = f"rank={RANK}, update_proj_gap={UPDATE_PROJ_GAP}, scale={SCALE}, prof_type={PROJ_TYPE}",  # Pass GaLore hyperparameters
        optim_target_modules=OPTIM_TARGET_MODULES,
        # GaLore-specific parameters
        # These are passed to the optimizer via TrainingArguments
        # rank, update_proj_gap, scale, proj_type are handled internally by galore-torch

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
        # criterion=nn.CrossEntropyLoss(),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[SystemMetricsCallback(log_dir=training_args.logging_dir, model=model, tokenizer=tokenizer)]
    )

    return trainer