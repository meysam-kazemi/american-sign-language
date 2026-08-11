"""Extract landmark feature vectors from the training image dataset and cache them.

Each sign class is saved to its own ``<label>.pkl`` containing a numpy array of
shape (n_samples, 63). Run as a script to (re)build the landmark cache.
"""

import os
import pickle

import cv2 as cv
import numpy as np

from ml.config import get_settings
from ml.landmarks import LandmarkDetector


def save_landmarks(features, path):
    """Save a numpy array of feature vectors to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(np.asarray(features, dtype=np.float32), f)


def load_saved_landmarks(settings=None):
    """Load all cached landmark features and labels.

    Returns:
        (X, y): X is an (N, 63) float32 array, y is a list of N string labels.
    """
    settings = settings or get_settings()
    landmarks_dir = settings.landmarks_dir
    all_features, all_labels = [], []

    if not os.path.isdir(landmarks_dir):
        raise RuntimeError(
            f"Landmark cache directory {landmarks_dir!r} does not exist. "
            "Run `python -m ml.preprocess` first."
        )

    print("\nLoading landmark data...")
    for pkl_file in sorted(f for f in os.listdir(landmarks_dir) if f.endswith(".pkl")):
        label = os.path.splitext(pkl_file)[0]
        with open(os.path.join(landmarks_dir, pkl_file), "rb") as f:
            features = pickle.load(f)
        all_features.append(np.asarray(features, dtype=np.float32))
        all_labels.extend([label] * len(features))
        print(f"  - Loaded {len(features)} samples for label '{label}'")

    X = np.concatenate(all_features, axis=0) if all_features else np.empty((0, 63), np.float32)
    print("Data loading complete.")
    return X, all_labels


def build_cache(settings=None):
    """Extract landmarks from every training image and write one pickle per sign."""
    settings = settings or get_settings()
    train_dir = settings.train_dir
    landmarks_dir = settings.landmarks_dir
    if not os.path.isdir(train_dir):
        raise RuntimeError(
            f"Training dataset directory {train_dir!r} does not exist. "
            "Download the ASL image dataset and configure ASL_TRAIN_DIR."
        )
    os.makedirs(landmarks_dir, exist_ok=True)

    # static_image_mode=True is correct for processing independent still images.
    detector = LandmarkDetector.from_settings(settings, static_image_mode=True)

    for sign_name in sorted(os.listdir(train_dir)):
        sign_dir = os.path.join(train_dir, sign_name)
        if not os.path.isdir(sign_dir):
            continue

        features = []  # reset per sign so each pickle holds only its own samples
        image_paths = sorted(os.listdir(sign_dir))
        if not image_paths:
            print(f"Skipping empty class directory {sign_dir!r}")
            continue
        for i, image_path in enumerate(image_paths, 1):
            img = cv.imread(os.path.join(sign_dir, image_path), 1)
            if img is None:
                continue
            vec = detector.extract_features(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            if vec is not None:
                features.append(vec)
            pct = 100 * i / len(image_paths)
            print(f"{sign_name:<8}: {pct:5.1f}%", end="\r")

        out_path = os.path.join(landmarks_dir, f"{sign_name}.pkl")
        save_landmarks(features, out_path)
        print(f"Saved {len(features):>5} samples for '{sign_name}' -> {out_path}")

    detector.close()


if __name__ == "__main__":
    build_cache()
