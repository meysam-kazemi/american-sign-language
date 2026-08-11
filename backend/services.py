"""Lazily-instantiated ML singletons shared across requests.

Heavy objects (the classifier, MediaPipe detector, Whisper, Piper) are created
on first use so the server starts fast and only loads what each feature needs.
"""

import os
from contextlib import suppress
from functools import lru_cache
from importlib.util import find_spec
from threading import Lock

from ml.config import get_settings


class ServiceUnavailableError(RuntimeError):
    """Raised when an optional model or runtime asset is unavailable."""


_inference_lock = Lock()


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
        raise ServiceUnavailableError(
            "ASL classifier is unavailable. Run preprocessing and training, then place "
            f"the model at {settings.model_path!r} and labels at {settings.labels_path!r}."
        )
    try:
        return SignClassifier(settings.model_path, settings.labels_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ServiceUnavailableError(f"Could not load the ASL classifier: {exc}") from exc


@lru_cache
def get_sign_images():
    from ml.gif_builder import load_sign_images

    settings = get_settings()
    try:
        images = load_sign_images(settings.signs_dir)
    except OSError as exc:
        raise ServiceUnavailableError(
            f"Sign images are unavailable at {settings.signs_dir!r}."
        ) from exc
    if not images:
        raise ServiceUnavailableError(f"No sign images were found below {settings.signs_dir!r}.")
    return images


@lru_cache
def get_stt():
    from ml.stt import SpeechToText

    try:
        return SpeechToText()
    except Exception as exc:
        raise ServiceUnavailableError(f"Could not load the Whisper model: {exc}") from exc


@lru_cache
def get_tts():
    from ml.tts import TextToSpeech

    settings = get_settings()
    missing = [
        path
        for path in (settings.tts_model_path, settings.tts_config_path)
        if not os.path.isfile(path)
    ]
    if missing:
        raise ServiceUnavailableError(
            "Piper voice files are unavailable: " + ", ".join(repr(path) for path in missing)
        )
    try:
        return TextToSpeech(settings.tts_model_path)
    except Exception as exc:
        raise ServiceUnavailableError(f"Could not load the Piper voice: {exc}") from exc


def predict_sign(frame_bgr):
    """Return a classifier prediction for one BGR frame, or ``None`` if no hand is found."""
    import cv2

    with _inference_lock:
        features = get_detector().extract_features(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if features is None:
            return None
        return get_classifier().predict(features)


def capability_status():
    """Return lightweight readiness information without loading heavy models."""
    settings = get_settings()
    model_ready = os.path.isfile(settings.model_path) and os.path.isfile(settings.labels_path)

    try:
        sign_images = get_sign_images()
        gif_ready = True
        gif_detail = f"{len(sign_images)} sign images available"
    except ServiceUnavailableError as exc:
        gif_ready = False
        gif_detail = str(exc)

    stt_ready = find_spec("faster_whisper") is not None
    tts_missing = [
        path
        for path in (settings.tts_model_path, settings.tts_config_path)
        if not os.path.isfile(path)
    ]

    return {
        "recognition": {
            "available": model_ready,
            "detail": (
                "classifier artifacts available"
                if model_ready
                else "train the classifier to create the model and labels artifacts"
            ),
        },
        "text_to_gif": {"available": gif_ready, "detail": gif_detail},
        "speech_to_text": {
            "available": stt_ready,
            "detail": (
                "Whisper runtime installed; the configured model loads on first use"
                if stt_ready
                else "install the backend dependency group"
            ),
        },
        "text_to_speech": {
            "available": not tts_missing,
            "detail": (
                "Piper voice files available"
                if not tts_missing
                else "missing " + ", ".join(tts_missing)
            ),
        },
    }


def clear_service_caches():
    """Release native resources and clear cached service objects."""
    if get_detector.cache_info().currsize:
        with suppress(Exception):
            get_detector().close()
    for factory in (get_detector, get_classifier, get_sign_images, get_stt, get_tts):
        factory.cache_clear()
