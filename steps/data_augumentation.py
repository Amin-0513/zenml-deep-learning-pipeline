import logging
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,f1_score
import numpy as np  
from zenml import step

@step
def data_augmentation_step(
    data_dir: str,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Loads data, applies transforms, and returns dataloaders."""

    # Training transforms (augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Datasets
    train_data = datasets.ImageFolder(data_dir, transform=train_transform)
    test_data = datasets.ImageFolder(data_dir, transform=test_transform)

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    class_names = train_data.classes

    return train_loader, test_loader, class_names
