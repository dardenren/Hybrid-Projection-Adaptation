import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import Trainer, TrainingArguments
from config import Config_Args, DEVICE, logger
from metrics import SystemMetricsCallback
from peft import LoraConfig, get_peft_model
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
# def compute_metrics(eval_pred):
    # predictions, labels = eval_pred
    # # Convert NumPy arrays to PyTorch Tensors
    # predictions = torch.tensor(predictions, device=DEVICE)
    # labels = torch.tensor(labels, device=DEVICE)
    # # Compute loss
    # loss = nn.CrossEntropyLoss()(predictions, labels)
    # mcc = matthews_corrcoef(labels, predictions)

    # return {
    #     "cross_entropy_loss": loss.item(),
    #     "mcc": mcc }

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    # optimizer = optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],  # Only optimize LoRA parameters
    #     lr=Config_Args.args.lr,      
    #     # betas=(0.9, 0.999),    
    #     # eps=1e-8,             
    #     # weight_decay=0.01,     
    #     # amsgrad=False         
    # )
    target_modules = [name for name, module in model.named_modules() if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))]
    config = LoraConfig(
        r=Config_Args.args.rank,
        lora_alpha=Config_Args.args.scale,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        # modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
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
        # max_grad_norm=10.0,
        optim="adamw_torch",
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, None),
        callbacks=[SystemMetricsCallback(log_dir=training_args.logging_dir, model=model, tokenizer=tokenizer)]
    )

    return trainer