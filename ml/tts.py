"""Text-to-speech using Piper (local, open-source). Produces WAV audio bytes."""
import io
import wave

from ml.config import get_settings


class TextToSpeech:
    def __init__(self, voice=None):
        from piper.voice import PiperVoice  # lazy: only needed when used

        self.voice_name = voice or get_settings().tts_voice
        # Piper expects a downloaded voice model (<voice>.onnx) on the model path.
        self.voice = PiperVoice.load(f"{self.voice_name}.onnx")

    def synthesize(self, text):
        """Synthesize `text` and return WAV-encoded audio bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self.voice.synthesize(text, wav)
        return buf.getvalue()
