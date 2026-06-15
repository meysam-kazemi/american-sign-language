"""Train the landmark MLP classifier on the cached landmark features."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.classifier import LandmarkMLP, save_labels
from ml.config import load_config
from ml.preprocess import load_saved_landmarks

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_landmark_mlp.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")


def train(config=None):
    config = config or load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.getint("TRAIN", "batch_size", fallback=32)
    num_epochs = config.getint("TRAIN", "epoch", fallback=100)

    X, y = load_saved_landmarks(config)
    if len(X) == 0:
        raise RuntimeError("No landmark data found. Run `python -m ml.preprocess` first.")

    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in y], dtype=np.int64)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx))
    val_size = max(1, int(0.15 * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print(f"Classes: {len(classes)} | Samples: {len(dataset)} | Device: {device}")

    model = LandmarkMLP(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        acc = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch:>3}/{num_epochs} | loss {running/len(train_loader):.4f} | val acc {acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            save_labels(classes, LABELS_PATH)

    print(f"Done. Best val acc {best_acc:.3f}. Saved -> {MODEL_PATH}")


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = torch.argmax(model(inputs), dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


if __name__ == "__main__":
    train()
