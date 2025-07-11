import torch.nn as nn
from transformers import Trainer, TrainingArguments
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, OPTIM, OPTIM_TARGET_MODULES, PROJ_TYPE, RANK, SCALE, SEED, UPDATE_PROJ_GAP

class CustomTrainer(Trainer):
    def __init__(self, criterion=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Criterion -> loss function
        """
        # Extract inputs and labels
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss

def setup_trainer(model, train_dataset, test_dataset):
    """Set up the Hugging Face Trainer with GaLore optimizer."""
    training_args = TrainingArguments(
        output_dir="output/fine_tuned_bert_galore.pt",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        optim=OPTIM,
        optim_args = f"rank={RANK}, update_proj_gap={UPDATE_PROJ_GAP}, scale={SCALE}, prof_type={PROJ_TYPE}",  # Pass GaLore hyperparameters
        optim_target_modules=OPTIM_TARGET_MODULES,
        # GaLore-specific parameters
        # These are passed to the optimizer via TrainingArguments
        # rank, update_proj_gap, scale, proj_type are handled internally by galore-torch

        logging_dir="output/training.log",
        logging_steps=100,  # Log every 100 steps
        logging_strategy="steps",
        save_strategy="no", # Set to "epoch" to save model after every epoch
        eval_strategy="epoch",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        criterion=nn.CrossEntropyLoss(),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    return trainer