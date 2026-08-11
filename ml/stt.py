"""Speech-to-text using faster-whisper (local, open-source)."""

from ml.config import get_settings


class SpeechToText:
    def __init__(self, model_size=None, device=None, compute_type=None):
        from faster_whisper import WhisperModel  # lazy: only needed when used

        settings = get_settings()
        model_size = model_size or settings.whisper_model
        device = device or settings.whisper_device
        compute_type = compute_type or settings.whisper_compute_type
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language="en"):
        """Transcribe an audio file to a single text string."""
        segments, _ = self.model.transcribe(audio_path, language=language)
        return " ".join(segment.text.strip() for segment in segments).strip()
