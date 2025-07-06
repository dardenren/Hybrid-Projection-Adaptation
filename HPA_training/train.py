import torch
from tqdm import trange
from config import *
import os
import time
from typing import *
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from hpa import HpaModule

class HpaTrainer():
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer, # Crucial: Optimizer instance
        criterion: nn.Module, # Crucial: Loss function instance (e.g., nn.CrossEntropyLoss)
        train_dataset: Dataset,
        eval_dataset: Dataset,
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        num_train_epochs: int = 3,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1, # Crucial: For accumulating gradients
        refresh_adapter_steps: int = 100, # Recalculates adapters every cycle
        refresh_type: Literal["weight", "gradient", "other"] = "gradient",
        rank_reduction_cycles: Optional[int] = None, # Every cycle (refresh_adapter_steps), find intrinsic rank (thus requiring SVD)
        first_rank_reduction_cycle: Optional[int] = None,
        output_dir: str = './results',
        logging_dir: str = './logs',
        logging_steps: int = 10,
        device: torch.device = None, # Crucial: Device to run training on (CPU/GPU)
        lr_scheduler: optim.lr_scheduler = None, # Crucial: Learning rate scheduler (can be built internally)
        max_grad_norm: float = 1.0 # Crucial: For gradient clipping
    ):
        # --- Model and Core Components ---
        self.model = model
        self.hpa_modules = [*filter(lambda x: isinstance(x, HpaModule), model.modules())]
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
        self.optimizer = optimizer
        self.criterion = criterion

        # --- Device Management ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # --- Training Arguments ---
        if refresh_adapter_steps % gradient_accumulation_steps != 0:
            raise ValueError("refresh_adapter_steps must be a multiple of gradient_accumulation_steps")

        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_training_steps = len(self.train_dataloader) * num_train_epochs
        if lr_scheduler is None:
            self.lr_scheduler = self._create_default_lr_scheduler()
        else:
            self.lr_scheduler = lr_scheduler

        self.refresh_adapter_steps = refresh_adapter_steps
        self.refresh_type = refresh_type
        if rank_reduction_cycles is None:
            self.rank_reduction_steps = self.num_training_steps
        else:
            self.rank_reduction_steps = rank_reduction_cycles * refresh_adapter_steps
        
        if first_rank_reduction_cycle is None:
            self.first_rank_reduction_step = self.rank_reduction_steps
        else:
            self.first_rank_reduction_step = first_rank_reduction_cycle * refresh_adapter_steps

        self.max_grad_norm = max_grad_norm

        # --- Logging and Checkpointing Setup ---
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        # os.makedirs(self.output_dir, exist_ok=True)
        # os.makedirs(self.logging_dir, exist_ok=True)
        self.global_step = 0

        print(f"Trainer initialized. Device: {self.device}")
        print(f"Total training steps: {self.num_training_steps}")

    def _create_default_lr_scheduler(self):
        """Creates a simple linear warmup then linear decay scheduler."""
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Linear decay
            return max(
                0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.warmup_steps))
            )
        return LambdaLR(self.optimizer, lr_lambda)

    def _train_epoch(self):
        self.model.train() # Set model to training mode
        total_loss = 0
        steps_since_last_log = 0

        for step, batch in enumerate(self.train_dataloader):

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            inputs = batch.get('input_ids') if 'input_ids' in batch else batch[0]
            labels = batch.get('labels') if 'labels' in batch else batch[1]

            if self.global_step % self.refresh_adapter_steps == 0:
                print(f"--- Refresh adapter at {self.global_step} ---")
                rank_reduction_condition = (self.global_step >= self.first_rank_reduction_step) and \
                (self.global_step - self.first_rank_reduction_step) % self.rank_reduction_steps == 0
                
                self.optimizer.merge_states()
                for module in self.hpa_modules:
                    module.merge_weights()

                if self.refresh_type != "weight" or rank_reduction_condition:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.logits, labels) # Extract logits
                    loss.backward()

                for module in self.hpa_modules:
                    if self.refresh_type == "weight":
                        module.reactivate_on_weight(rank_reduction_condition)
                    elif self.refresh_type == "gradient":
                        module.reactivate_on_gradient(rank_reduction_condition)
                    else:
                        pass 

                self.optimizer.project_states()

            outputs = self.model(inputs)
            loss = self.criterion(outputs.logits, labels) # Extract logits
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (self.global_step % self.gradient_accumulation_steps == 0) or (self.global_step == self.num_training_steps): # Last step

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step() # Update learning rate
                self.optimizer.zero_grad() # Clear gradients

            total_loss += loss.item() * self.gradient_accumulation_steps # Scale back loss for logging

            self.global_step += 1
            steps_since_last_log += 1

            # --- Logging ---
            if self.global_step % self.logging_steps == 0:
                avg_loss = total_loss / steps_since_last_log
                current_lr = self.lr_scheduler.get_last_lr()[0]
                print(f"Step {self.global_step}/{self.num_training_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {datetime.now().strftime('%H:%M:%S')}")

                total_loss = 0 # Reset for next logging interval
                steps_since_last_log = 0

    def _evaluate(self, epoch=0):
        self.model.eval() # Set model to evaluation mode
        eval_loss = 0
        correct_predictions = 0
        total_samples = 0

        print(f"\nEvaluating after epoch {epoch + 1}...")
        with torch.no_grad(): # Disable gradient computation during evaluation
            for step, batch in enumerate(self.eval_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                inputs = batch.get('input_ids') if 'input_ids' in batch else batch[0]
                labels = batch.get('labels') if 'labels' in batch else batch[1]

                outputs = self.model(inputs)

                loss = self.criterion(outputs.logits, labels) # Extract logits
                eval_loss += loss.item()

                # Example metric: Accuracy for classification
                if outputs.logits.ndim > 1 and outputs.logits.shape[1] > 1: # Assuming classification
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                else: # Assuming regression or binary classification, adjust as needed
                    pass # No generic accuracy calculation without knowing task

                total_samples += labels.numel() # Count total labels for accuracy denominator

        avg_eval_loss = eval_loss / len(self.eval_dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        print(f"Evaluation Results (Epoch {epoch + 1}):")
        print(f"  Avg Eval Loss: {avg_eval_loss:.4f}")
        if total_samples > 0:
            print(f"  Accuracy: {accuracy:.4f}")

        self.model.train() # Set model back to training mode
        return avg_eval_loss, accuracy

    # def _save_checkpoint(self, epoch):
    #     checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
    #     torch.save({
    #         'epoch': epoch,
    #         'global_step': self.global_step,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
    #         # 'best_metric': self.best_metric, # If you implement early stopping
    #     }, checkpoint_path)
    #     print(f"Checkpoint saved to {checkpoint_path}")

    def train(self):
        print("Starting training...")
        start_time = time.time()

        for epoch in range(self.num_train_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_train_epochs} ---")
            self._train_epoch()

            # Evaluate after each epoch
            self._evaluate(epoch)

            # Save checkpoint after each epoch
            # self._save_checkpoint(epoch)

        end_time = time.time()
        print(f"\nTraining complete! Total time: {end_time - start_time:.2f} seconds")
  





