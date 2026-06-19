"""Speech-to-text using faster-whisper (local, open-source)."""
from ml.config import get_settings


class SpeechToText:
    def __init__(self, model_size=None, device="auto", compute_type="int8"):
        from faster_whisper import WhisperModel  # lazy: only needed when used

        model_size = model_size or get_settings().whisper_model
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language="en"):
        """Transcribe an audio file to a single text string."""
        segments, _ = self.model.transcribe(audio_path, language=language)
        return " ".join(segment.text.strip() for segment in segments).strip()
