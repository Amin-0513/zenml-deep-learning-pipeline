import logging
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from zenml import step
import socket
from datetime import datetime

# ------------------ Model ------------------
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ------------------ Helper: send file ------------------
def send_file(model_path: str, username: str, host='127.0.0.1', port=5001):
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(client_socket.recv(1024).decode())  # "Connection established"

    # Prepare metadata
    filename = os.path.basename(model_path)
    filesize = os.path.getsize(model_path)
    metadata = {
        "username": username,
        "filename": filename,
        "filesize": filesize
    }

    # Send metadata as JSON
    client_socket.sendall(json.dumps(metadata).encode())

    # Send file bytes
    with open(model_path, "rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            client_socket.sendall(data)

    print(f"Sent {filename} successfully")
    # client_socket.sendall(b"close")  # Tell server to close connection
    # client_socket.close()
    print("Connection closed")

# ------------------ ZenML Step ------------------
@step(enable_cache=False)
def model_training_step(
    username: str,
    model_path: str,
    train_loader: DataLoader,
    device: str,
    epochs: int = 200
) -> Tuple[nn.Module, List[float], List[float], List[float]]:

    """Trains a CNN model and sends it to the server."""
    model = BrainTumorCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(100 * correct / total)

        if (epoch + 1) % 20 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

    # Save model
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved at {model_path}")
    print(f"Model training completed and saved at {model_path}")

    # Send model to server
    send_file(model_path, username)

    return model, train_loss_list, train_acc_list, val_acc_list
