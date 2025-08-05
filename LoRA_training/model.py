import torch
import torch.nn as nn
from lora import *
from config import logger, Config_Args
from helper import TASK_TO_LABELS

from transformers import AutoModelForSequenceClassification

# Add LoRA matrices to the model
def set_module(model, module_name, new_module):
    module_path = module_name.split('.')
    current_module = model
    for part in module_path[:-1]:
        if part.isdigit():
            current_module = current_module[int(part)]
        else:
            current_module = getattr(current_module, part)
    last_part = module_path[-1]
    setattr(current_module, last_part, new_module)


def load_model(model_name, r, lora_alpha, merge_weights=True, linear=True, embedding=True):
    """
    linear : bool -> Option to change Linear layer to LoRALinear
    embedding : bool -> Option to change Embedding layer to LoRAEmbedding
    """
    logger.info(f"Initializing model: {Config_Args.args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(Config_Args.args.model_name, num_labels=TASK_TO_LABELS[Config_Args.args.task_name])                                                             
    logger.info("Model initialized")
    if not linear and not embedding:
        return model
    
    if linear:
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                set_module(model, name, LoRALinear(module.in_features, module.out_features, r=r, lora_alpha=lora_alpha))
            
    if embedding:
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Embedding):
                set_module(model, name, LoRAEmbedding(  
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    r=r,
                    lora_alpha=lora_alpha,
                    padding_idx=module.padding_idx
                )
            )
    logger.info("Model initialized and adapted to LoRA")
    return model

def merge_lora_weights(model):
    """
    Manually merge LoRA weights into the base model's weights for all LoRA layers.
    
    Args:
        model: The PyTorch model containing LoRA layers.
    """
    for module in model.modules():
        if isinstance(module, LoRALayer) and module.merge_weights and not module.merged:
            if module.r > 0:  # Ensure LoRA is active
                if isinstance(module, LoRALinear):
                    # Handle fan_in_fan_out transposition
                    def T(w):
                        return w.transpose(0, 1) if module.fan_in_fan_out else w
                    update = T(module.lora_B @ module.lora_A) * module.scaling
                    module.weight.data += update
                    logger.info(f"M")
                elif isinstance(module, LoRAEmbedding):
                    update = (module.lora_B @ module.lora_A).transpose(0, 1) * module.scaling
                    module.weight.data += update
                elif isinstance(module, ConvLoRA):
                    update = (module.lora_B @ module.lora_A).view(module.conv.weight.shape) * module.scaling
                    module.conv.weight.data += update
                elif isinstance(module, LoRAMergedLinear):
                    def T(w):
                        return w.transpose(0, 1) if module.fan_in_fan_out else w
                    update = module.merge_AB() * module.scaling
                    module.weight.data += update
                module.merged = True  # Mark as merged to prevent re-merging






