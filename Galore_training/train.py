import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import Trainer, TrainingArguments
from config import Config_Args, OPTIM, OPTIM_TARGET_MODULES, DEVICE
from metrics import SystemMetricsCallback
from galore_torch import GaLoreAdamW
from sklearn.metrics import matthews_corrcoef

def compute_metrics(eval_pred):
    predictions_np, labels_np = eval_pred
    predicted_labels_np = np.argmax(predictions_np, axis=1)
    mcc = matthews_corrcoef(labels_np, predicted_labels_np)
    
    # Compute loss if needed
    predictions_tensor = torch.tensor(predictions_np, device=DEVICE)
    labels_tensor = torch.tensor(labels_np, device=DEVICE)
    loss = nn.CrossEntropyLoss()(predictions_tensor, labels_tensor)
    
    return {
        "cross_entropy_loss": loss.item(),
        "mcc": mcc
    }

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    output_dir_string = "./output" + f"_{Config_Args.args.task_name}"
    logging_dir_string = "./logs" + f"_{Config_Args.args.task_name}"

    galore_params = [p for n, p in model.named_parameters() if len(p.shape) == 2 and ("dense" in n or "query" in n or "key" in n or "value" in n)]
    non_galore_params = [p for n, p in model.named_parameters() if not ("dense" in n or "query" in n or "key" in n or "value" in n)]
    param_groups = [
        {'params': non_galore_params},
        {'params': galore_params, 'rank': Config_Args.args.rank, 'update_proj_gap': Config_Args.args.proj_freq, 'scale': Config_Args.args.scale, 'proj_type': Config_Args.args.proj_type}
        ]
    optimizer = GaLoreAdamW(param_groups, lr=Config_Args.args.lr)

    training_args = TrainingArguments(
        # output_dir="./output",
        output_dir=output_dir_string,
        num_train_epochs=Config_Args.args.epochs,
        per_device_train_batch_size=Config_Args.args.train_batch_size,
        learning_rate=Config_Args.args.lr,
        warmup_ratio = 0.1,
        lr_scheduler_type="linear",
        seed=Config_Args.args.seed,
        # optim=OPTIM,
        # optim_args = f"rank={Config_Args.args.rank}, update_proj_gap={Config_Args.args.proj_freq}, scale={Config_Args.args.scale}, proj_type={Config_Args.args.proj_type}",  # Pass GaLore hyperparameters
        # optim_target_modules=OPTIM_TARGET_MODULES,

        # GaLore-specific parameters
        # These are passed to the optimizer via TrainingArguments
        # rank, update_proj_gap, scale, proj_type are handled internally by galore-torch

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
        # criterion=nn.CrossEntropyLoss(),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[SystemMetricsCallback(log_dir=training_args.logging_dir, model=model, tokenizer=tokenizer)]
    )

    return trainer