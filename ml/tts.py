"""Text-to-speech using Piper (local, open-source). Produces WAV audio bytes."""

import io
import wave

from ml.config import get_settings


class TextToSpeech:
    def __init__(self, model_path=None):
        from piper.voice import PiperVoice  # lazy: only needed when used

        self.model_path = model_path or get_settings().tts_model_path
        # Piper expects the ONNX model and a sibling ``.onnx.json`` config file.
        self.voice = PiperVoice.load(self.model_path)

    def synthesize(self, text):
        """Synthesize `text` and return WAV-encoded audio bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self.voice.synthesize(text, wav)
        return buf.getvalue()
