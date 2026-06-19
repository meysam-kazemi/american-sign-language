"""Lazily-instantiated ML singletons shared across requests.

Heavy objects (the classifier, MediaPipe detector, Whisper, Piper) are created
on first use so the server starts fast and only loads what each feature needs.
"""
import os
from functools import lru_cache

from fastapi import HTTPException

from ml.config import get_settings


@lru_cache
def get_detector():
    from ml.landmarks import LandmarkDetector

    # static_image_mode=False suits streamed video frames.
    return LandmarkDetector.from_settings(get_settings(), static_image_mode=False)


@lru_cache
def get_classifier():
    from ml.classifier import SignClassifier

    settings = get_settings()
    if not (os.path.exists(settings.model_path) and os.path.exists(settings.labels_path)):
        raise HTTPException(
            status_code=503,
            detail="ASL model not found. Train it and place artifacts in the models/ dir.",
        )
    return SignClassifier(settings.model_path, settings.labels_path)


@lru_cache
def get_sign_images():
    from ml.gif_builder import load_sign_images

    return load_sign_images()


@lru_cache
def get_stt():
    from ml.stt import SpeechToText

    return SpeechToText()


@lru_cache
def get_tts():
    from ml.tts import TextToSpeech

    return TextToSpeech()
