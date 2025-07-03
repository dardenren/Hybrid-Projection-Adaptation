import torch
import logging
from math import ceil
from tqdm import trange
from config import *


def train_model(model, dataset, epochs, batch_size, lr, optimizer, device, loss_fn="default"):
  model.train()

  # For custom loss function
  # loss_function = torch.nn.CrossEntropyLoss()

  logger.info(f"Starting training with epochs={epochs}, batch_size={batch_size}, lr={lr}")
  for epoch in range(epochs):
    total_loss = 0
    max_len = len(dataset)
    num_batches = ceil(max_len / batch_size)
    with trange(num_batches, desc="Training", unit="Batch Done") as pbar:
        for i in pbar:
            right_limit = (i+1)*batch_size if (i+1)*batch_size < max_len else max_len
            batch = dataset[i * batch_size:right_limit]
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {k: v.to(device) for k, v in dataset[i*batch_size:right_limit].items()}

                                                                                            
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward() # CrossEntropyLoss by default
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max_len
    logger.info(f"Epoch {epoch+1}/{epochs} completed, Average Training Loss: {avg_loss:.4f}")
    # print(f"Epoch {epoch + 1}/{EPOCHS}, Average Training Loss: {avg_loss:.4f}")

  





