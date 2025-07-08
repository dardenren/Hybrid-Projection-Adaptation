import torch
import torch.nn as nn
from lora import *
from config import logger, MODEL_NAME

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


def load_model(model_name=MODEL_NAME, r=4, lora_alpha=16, merge_weights=True, linear=True, embedding=True):
    """
    linear : bool -> Option to change Linear layer to LoRALinear
    embedding : bool -> Option to change Embedding layer to LoRAEmbedding
    """
    logger.info(f"Initializing model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) 
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






