"""Flow 3 (voice -> text -> GIF) and Flow 4 helper (text -> speech)."""

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from backend.schemas import TextRequest, TextResponse
from backend.services import ServiceUnavailableError, get_sign_images, get_stt, get_tts
from ml.config import get_settings
from ml.gif_builder import MissingSignImagesError, build_gif

router = APIRouter(prefix="/api", tags=["voice"])


_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}
_UPLOAD_CHUNK_SIZE = 1024 * 1024


async def _save_upload(upload: UploadFile) -> str:
    """Stream an audio upload to a temporary file while enforcing a size limit."""
    content_type = (upload.content_type or "").lower()
    if content_type and not (
        content_type.startswith("audio/")
        or content_type in {"application/octet-stream", "video/webm"}
    ):
        raise HTTPException(status_code=415, detail="The uploaded file must contain audio.")

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        suffix = ".audio"

    limit = get_settings().max_upload_bytes
    size = 0
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as destination:
            path = destination.name
            while chunk := upload.file.read(_UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio upload exceeds the {limit // (1024 * 1024)} MiB limit.",
                    )
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="The uploaded audio file is empty.")
        return path
    except Exception:
        if path:
            with suppress(FileNotFoundError):
                os.unlink(path)
        raise


@router.post("/speech-to-text", response_model=TextResponse)
async def speech_to_text(audio: Annotated[UploadFile, File()]):
    """Transcribe an uploaded audio file to text (Whisper)."""
    path = await _save_upload(audio)
    try:
        try:
            text = await asyncio.to_thread(get_stt().transcribe, path)
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Audio could not be processed: {exc}"
            ) from exc
        if not text:
            raise HTTPException(status_code=422, detail="No speech could be transcribed.")
        return TextResponse(text=text)
    finally:
        os.unlink(path)


@router.post("/voice-to-gif")
async def voice_to_gif(audio: Annotated[UploadFile, File()]):
    """Transcribe spoken audio, then return an ASL GIF of the recognized text."""
    path = await _save_upload(audio)
    try:
        try:
            text = await asyncio.to_thread(get_stt().transcribe, path)
        except ServiceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Audio could not be processed: {exc}"
            ) from exc
    finally:
        os.unlink(path)
    if not text:
        raise HTTPException(status_code=422, detail="No speech could be transcribed.")
    try:
        data = build_gif(text, sign_images=get_sign_images(), strict=True)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MissingSignImagesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if data is None:
        raise HTTPException(status_code=422, detail=f"No signs mapped for: {text!r}")
    return Response(
        content=data,
        media_type="image/gif",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/text-to-speech")
async def text_to_speech(req: TextRequest):
    """Synthesize speech (WAV) from text (Piper). Used by Flow 4."""
    try:
        data = await asyncio.to_thread(get_tts().synthesize, req.text)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Speech synthesis failed: {exc}") from exc
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )
