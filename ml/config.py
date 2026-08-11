"""Application settings loaded from environment variables or a local ``.env`` file."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env at import time if present


@dataclass(frozen=True)
class Settings:
    # Data locations
    train_dir: str
    test_dir: str
    landmarks_dir: str
    model_dir: str
    signs_dir: str
    # Landmark detection
    landmark_min_confidence: float
    landmark_num_hands: int
    # Training
    train_batch_size: int
    train_epochs: int
    train_seed: int
    # Stream decoding
    decode_stability: int
    decode_min_confidence: float
    # Backend
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]
    max_upload_bytes: int
    max_frame_bytes: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    tts_model_path: str

    @property
    def model_path(self):
        return os.path.join(self.model_dir, "asl_landmark_mlp.pth")

    @property
    def labels_path(self):
        return os.path.join(self.model_dir, "labels.json")

    @property
    def tts_config_path(self):
        return f"{self.tts_model_path}.json"


def _positive_int(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _probability(name, default):
    value = float(os.getenv(name, str(default)))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _origins():
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@lru_cache
def get_settings():
    train_dir = os.getenv("ASL_TRAIN_DIR", "dataset/asl_alphabet_train")
    tts_voice = os.getenv("TTS_VOICE", "en_US-amy-medium")
    return Settings(
        train_dir=train_dir,
        test_dir=os.getenv("ASL_TEST_DIR", "dataset/asl_alphabet_test"),
        landmarks_dir=os.getenv("ASL_LANDMARKS_DIR", "dataset/landmarks"),
        model_dir=os.getenv("ASL_MODEL_DIR", "models"),
        signs_dir=os.getenv("ASL_SIGNS_DIR", train_dir),
        landmark_min_confidence=_probability("LANDMARK_MIN_CONFIDENCE", 0.3),
        landmark_num_hands=_positive_int("LANDMARK_NUM_HANDS", 1),
        train_batch_size=_positive_int("TRAIN_BATCH_SIZE", 32),
        train_epochs=_positive_int("TRAIN_EPOCHS", 100),
        train_seed=int(os.getenv("TRAIN_SEED", "42")),
        decode_stability=_positive_int("DECODE_STABILITY", 8),
        decode_min_confidence=_probability("DECODE_MIN_CONFIDENCE", 0.6),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_positive_int("API_PORT", 8000),
        cors_origins=_origins(),
        max_upload_bytes=_positive_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
        max_frame_bytes=_positive_int("MAX_FRAME_BYTES", 2 * 1024 * 1024),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        tts_model_path=os.getenv("TTS_MODEL_PATH", str(Path("voices") / f"{tts_voice}.onnx")),
    )
