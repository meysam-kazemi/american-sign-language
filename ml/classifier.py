"""MLP classifier that maps a 63-dim hand-landmark vector to an ASL sign class."""

import json
import os

import numpy as np
import torch
import torch.nn as nn


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes, input_dim=63):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SignClassifier:
    """Inference wrapper: loads a trained MLP plus its label map."""

    def __init__(self, model_path, labels_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(labels_path) as f:
            self.labels = json.load(f)  # list: index -> label string
        if (
            not isinstance(self.labels, list)
            or not self.labels
            or not all(isinstance(label, str) and label for label in self.labels)
        ):
            raise ValueError("labels.json must contain a non-empty list of label strings")
        self.model = LandmarkMLP(num_classes=len(self.labels))
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def predict(self, feature_vector):
        """Return (label, confidence) for a single 63-dim feature vector."""
        features = np.asarray(feature_vector, dtype=np.float32)
        if features.shape != (63,):
            raise ValueError(f"Expected 63 landmark features, received shape {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("Landmark features must contain only finite values")
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(torch.argmax(probs))
        return self.labels[idx], float(probs[idx])


def save_labels(labels, path):
    labels = list(labels)
    if not labels or not all(isinstance(label, str) and label for label in labels):
        raise ValueError("labels must be a non-empty sequence of strings")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f)
