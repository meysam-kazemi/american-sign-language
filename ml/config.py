"""Application settings loaded from environment variables / .env file."""
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # read .env at import time if present


@dataclass(frozen=True)
class Settings:
    # Data locations
    train_dir: str
    test_dir: str
    landmarks_dir: str
    model_dir: str
    # Landmark detection
    landmark_min_confidence: float
    landmark_num_hands: int
    # Training
    train_batch_size: int
    train_epochs: int
    # Stream decoding
    decode_stability: int
    decode_min_confidence: float
    # Backend
    api_host: str
    api_port: int
    whisper_model: str
    tts_voice: str

    @property
    def model_path(self):
        return os.path.join(self.model_dir, "asl_landmark_mlp.pth")

    @property
    def labels_path(self):
        return os.path.join(self.model_dir, "labels.json")


@lru_cache
def get_settings():
    return Settings(
        train_dir=os.getenv("ASL_TRAIN_DIR", "dataset/asl_alphabet_train"),
        test_dir=os.getenv("ASL_TEST_DIR", "dataset/asl_alphabet_test"),
        landmarks_dir=os.getenv("ASL_LANDMARKS_DIR", "dataset/landmarks"),
        model_dir=os.getenv("ASL_MODEL_DIR", "models"),
        landmark_min_confidence=float(os.getenv("LANDMARK_MIN_CONFIDENCE", "0.3")),
        landmark_num_hands=int(os.getenv("LANDMARK_NUM_HANDS", "2")),
        train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "32")),
        train_epochs=int(os.getenv("TRAIN_EPOCHS", "100")),
        decode_stability=int(os.getenv("DECODE_STABILITY", "8")),
        decode_min_confidence=float(os.getenv("DECODE_MIN_CONFIDENCE", "0.6")),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        tts_voice=os.getenv("TTS_VOICE", "en_US-amy-medium"),
    )
