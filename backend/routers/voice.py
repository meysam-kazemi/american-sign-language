"""Flow 3 (voice -> text -> GIF) and Flow 4 helper (text -> speech)."""
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from backend.schemas import TextRequest, TextResponse
from backend.services import get_sign_images, get_stt, get_tts
from ml.gif_builder import build_gif

router = APIRouter(prefix="/api", tags=["voice"])


async def _save_upload(upload: UploadFile, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(await upload.read())
    return path


@router.post("/speech-to-text", response_model=TextResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file to text (Whisper)."""
    path = await _save_upload(audio, os.path.splitext(audio.filename or "")[1] or ".wav")
    try:
        return TextResponse(text=get_stt().transcribe(path))
    finally:
        os.unlink(path)


@router.post("/voice-to-gif")
async def voice_to_gif(audio: UploadFile = File(...)):
    """Transcribe spoken audio, then return an ASL GIF of the recognized text."""
    path = await _save_upload(audio, os.path.splitext(audio.filename or "")[1] or ".wav")
    try:
        text = get_stt().transcribe(path)
    finally:
        os.unlink(path)
    data = build_gif(text, sign_images=get_sign_images())
    if data is None:
        raise HTTPException(status_code=422, detail=f"No signs mapped for: {text!r}")
    return Response(content=data, media_type="image/gif")


@router.post("/text-to-speech")
def text_to_speech(req: TextRequest):
    """Synthesize speech (WAV) from text (Piper). Used by Flow 4."""
    return Response(content=get_tts().synthesize(req.text), media_type="audio/wav")
