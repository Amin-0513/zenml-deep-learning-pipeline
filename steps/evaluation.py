import logging
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from zenml.client import Client
from zenml import step

from typing import Tuple

@step
def evaluate_model_step(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: str, 
    class_names: list[str]
) -> Tuple[float, float, float]:
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print(classification_report(y_true, y_pred, target_names=class_names))

    # Log metadata
    try:
        client = Client()
        client.log_artifact_metadata(
            artifact_name="local",
            metadata={
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
            }
        )
    except Exception as e:
        logging.warning(f"Could not log artifact metadata: {e}")

    logging.info(f"Model evaluation completed. Accuracy: {accuracy}")
    
    return accuracy, precision, recall

