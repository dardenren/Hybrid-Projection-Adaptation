import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from config import Config_Args, OPTIM_TARGET_MODULES, DEVICE, logger
from metrics import SystemMetricsCallback
from hpa import make_flexi_optimizer
from trainer import HpaTrainer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert NumPy arrays to PyTorch Tensors
    predictions = torch.tensor(predictions, device=DEVICE)
    labels = torch.tensor(labels, device=DEVICE)
    # Compute loss
    loss = nn.CrossEntropyLoss()(predictions, labels)
    return {"eval_loss": loss.item()}

def setup_trainer(model, tokenizer, train_dataset, test_dataset):
    output_dir_string = "./output" + f"_{Config_Args.args.task_name}"
    logging_dir_string = "./logs" + f"_{Config_Args.args.task_name}"

    # HPA optimizer
    identity_fn = lambda x: x
    sq_fn = lambda x: x*x
    sqrt_fn = lambda x: torch.sqrt(x)
    FsAdamW = make_flexi_optimizer(torch.optim.AdamW, {"exp_avg": (identity_fn, identity_fn), "exp_avg_sq": (sqrt_fn, sq_fn)})
    optimizer = FsAdamW(
        model,
        model.named_parameters(), 
        lr=Config_Args.args.lr
    )

    print(type(train_dataset))
    total_steps = int((len(train_dataset) / 32) * Config_Args.args.epochs)
    warmup_ratio = 0.1
    warmup_steps = int(total_steps * warmup_ratio)
    trainer = HpaTrainer(
            model=model,
            optimizer=optimizer,
            device = None, 
            criterion=torch.nn.CrossEntropyLoss(),
            output_dir=output_dir_string,
            logging_dir=logging_dir_string,
            # learning_rate=Config_Args.args.lr,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            train_batch_size=Config_Args.args.train_batch_size,
            eval_batch_size=Config_Args.args.train_batch_size,
            num_train_epochs=Config_Args.args.epochs,
            logging_steps = 100,
            warmup_steps = warmup_steps,
            # max_grad_norm = 1.0, 

            # HPA hyperparameters
            gradient_accumulation_steps = 1, 
            refresh_adapter_steps = Config_Args.args.proj_freq, 
            refresh_type = "gradient",
            rank_reduction_cycles = None,
            first_rank_reduction_cycle = None,


            compute_metrics=compute_metrics,
            seed=Config_Args.args.seed,
            report_to="tensorboard", 
            callbacks=[SystemMetricsCallback(log_dir=logging_dir_string, model=model, tokenizer=tokenizer)],

    )


    return trainer