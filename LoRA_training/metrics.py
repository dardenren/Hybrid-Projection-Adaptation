import time
import torch
import psutil
import pynvml # type: ignore
from config import logger
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

class SystemMetricsCallback(TrainerCallback):
    def __init__(self, log_dir="./logs", model=None, tokenizer=None, eval_batch_size=1):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.start_time = None
        self.step_start_time = None
        self.device_count = 0
        
        # Initialize pynvml for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            logger.info(f"Failed to load GPU: {e}")
        
        # Calculate number of trainable parameters (static)
        if model is not None:
            self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.writer.add_scalar('Model/Trainable_Parameters', self.trainable_params, 0)

            # Log the model graph
            try:
                # Create a dummy input for the model (compatible with transformers)
                if self.tokenizer is not None:
                    dummy_input = self.tokenizer(
                        "Sample text for graph visualization",
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(model.device)
                    # Log the graph using the model and dummy input
                    self.writer.add_graph(model, input_to_model=dummy_input)
                    logger.info("Model graph logged to TensorBoard")
                else:
                    logger.info("Tokenizer not provided; skipping model graph logging")
            except Exception as e:
                logger.info(f"Failed to log model graph: {e}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Record start time for total training time
        self.start_time = time.time()
        logger.info("Training started")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Record start time for each training step
        self.step_start_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, train_dataloader=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            # Calculate step time
            step_time = time.time() - self.step_start_time
            self.writer.add_scalar('Training/Step_Time (s)', step_time, state.global_step)

            # CPU usage
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1)
            except Exception as e:
                logger.info(f"CPU monitoring failed: {e}")
                cpu_usage = 0
            self.writer.add_scalar('System/CPU_Usage (%)', cpu_usage, state.global_step)

            # GPU usage
            if self.device_count != 0:
                for i in range(self.device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.writer.add_scalar(f'System/GPU_{i}_Utilization (%)', util.gpu, state.global_step)
                        self.writer.add_scalar(f'System/GPU_{i}_Memory_Used (MB)', mem.used / 1024 / 1024, state.global_step)
                    except Exception as e:
                        logger.info(f"GPU {i} monitoring failed: {e}")

            # Training throughput (tokens per second)
            if train_dataloader is not None:
                try:
                    batch = next(iter(train_dataloader))
                    input_ids = batch.get('input_ids', None)
                    if input_ids is not None:
                        total_tokens = input_ids.numel()  # batch_size * sequence_length
                        throughput_tokens = total_tokens / step_time if step_time > 0 else 0
                        self.writer.add_scalar('Training/Throughput (tokens/s)', throughput_tokens, state.global_step)
                    else:
                        logger.info("No input_ids; skipping token throughput")
                except Exception as e:
                    logger.info(f"Token throughput calculation failed: {e}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        # Log evaluation metrics (e.g., validation loss, perplexity)
        if metrics and "eval_loss" in metrics:
        
            try:
                eval_loss = metrics["eval_loss"]
                logger.info(f"Step {state.global_step}: Evaluation Loss: {eval_loss:.4f}")
                perplexity = torch.exp(torch.tensor(metrics["eval_loss"])).item()
                self.writer.add_scalar('Evaluation/Perplexity', perplexity, state.global_step)
            except Exception as e:
                logger.info(f"Perplexity or Accuracy calculation failed: {e}")

        # Log accuracy
        if metrics and "eval_accuracy" in metrics:
            try:
                eval_accuracy = metrics["eval_accuracy"]
                self.writer.add_scalar('Evaluation/Accuracy', eval_accuracy, state.global_step)
                logger.info(f"Step {state.global_step}: Evaluation Accuracy: {eval_accuracy:.4f}")
            except Exception as e:
                logger.info(f"Step {state.global_step}: Evaluation Accuracy not available")

        # Measure inference latency
        if self.model is not None and self.tokenizer is not None:
            self.model.eval()
            with torch.no_grad():
                try:
                    inputs = self.tokenizer("Sample text for inference", return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    start_time = time.time()
                    _ = self.model(**inputs)
                    inference_latency = time.time() - start_time
                    self.writer.add_scalar('Evaluation/Inference_Latency (s)', inference_latency, state.global_step)
                except Exception as e:
                    logger.info(f"Inference latency measurement failed: {e}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log total training time
        elapsed_time = time.time() - self.start_time
        self.writer.add_scalar('Training/Total_Time (s)', elapsed_time, state.global_step)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if hasattr(self, 'trainer') and self.trainer is not None:
            try:
                metrics = self.trainer.evaluate()
                self.logger.info("Final evaluation metrics:")
                if metrics:
                    if "eval_loss" in metrics:
                        self.writer.add_scalar('Final/Evaluation_Loss', metrics["eval_loss"], state.global_step)
                        logger.info(f"Final Evaluation Loss: {metrics['eval_loss']:.4f}")
                    if "eval_accuracy" in metrics:
                        self.writer.add_scalar('Final/Evaluation_Accuracy', metrics["eval_accuracy"], state.global_step)
                        logger.info(f"Final Evaluation Accuracy: {metrics['eval_accuracy']:.4f}")
                    if "eval_precision" in metrics:
                        self.writer.add_scalar('Final/Evaluation_Precision', metrics["eval_precision"], state.global_step)
                        logger.info(f"Final Evaluation Precision: {metrics['eval_precision']:.4f}")
                    if "eval_recall" in metrics:
                        self.writer.add_scalar('Final/Evaluation_Recall', metrics["eval_recall"], state.global_step)
                        logger.info(f"Final Evaluation Recall: {metrics['eval_recall']:.4f}")
                    if "eval_f1" in metrics:
                        self.writer.add_scalar('Final/Evaluation_F1', metrics["eval_f1"], state.global_step)
                        logger.info(f"Final Evaluation F1 Score: {metrics['eval_f1']:.4f}")
                    if not any(k in metrics for k in ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]):
                        logger.info("No classification metrics available in final evaluation")
            except Exception as e:
                print(f"Final evaluation failed: {e}")
                self.logger.info(f"Final evaluation failed: {e}")
        
        # Cleanup
        try:
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.info(f"pynvml shutdown failed: {e}")
        self.writer.close()
        logger.info("Training completed")