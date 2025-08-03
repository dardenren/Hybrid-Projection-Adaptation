import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import logger
from torch.utils.data import DataLoader

def evaluate_model(model, test_dataset):
    test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = []
    true_labels = []

    for batch in test_dataset:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1)

        predictions.extend(batch_predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")