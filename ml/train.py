"""Train the landmark MLP classifier on the cached landmark features."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.classifier import LandmarkMLP, save_labels
from ml.config import get_settings
from ml.preprocess import load_saved_landmarks


def train(settings=None):
    settings = settings or get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = settings.train_batch_size
    num_epochs = settings.train_epochs
    torch.manual_seed(settings.train_seed)
    np.random.seed(settings.train_seed)

    X, y = load_saved_landmarks(settings)
    if len(X) < 2:
        raise RuntimeError("No landmark data found. Run `python -m ml.preprocess` first.")

    classes = sorted(set(y))
    if len(classes) < 2:
        raise RuntimeError("Training requires landmark samples from at least two sign classes.")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in y], dtype=np.int64)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx))
    val_size = min(len(dataset) - 1, max(1, int(0.15 * len(dataset))))
    generator = torch.Generator().manual_seed(settings.train_seed)
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=generator,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print(f"Classes: {len(classes)} | Samples: {len(dataset)} | Device: {device}")

    model = LandmarkMLP(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    os.makedirs(settings.model_dir, exist_ok=True)
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
        loss = running / len(train_loader)
        print(f"Epoch {epoch:>3}/{num_epochs} | loss {loss:.4f} | val acc {acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), settings.model_path)
            save_labels(classes, settings.labels_path)

    print(f"Done. Best val acc {best_acc:.3f}. Saved -> {settings.model_path}")


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
