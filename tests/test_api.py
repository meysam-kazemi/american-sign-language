import io
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.routers import gif, voice
from backend.services import ServiceUnavailableError

pytestmark = pytest.mark.anyio


def _image(path):
    Image.new("RGB", (12, 12), "white").save(path)


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.2.0"}


async def test_status_reports_degraded_capabilities(client, monkeypatch):
    monkeypatch.setattr(
        "backend.main.capability_status",
        lambda: {"recognition": {"available": False, "detail": "missing model"}},
    )
    response = await client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


async def test_text_request_rejects_blank_text(client):
    response = await client.post("/api/text-to-gif", json={"text": "   "})
    assert response.status_code == 422


async def test_text_to_gif_returns_an_image(client, monkeypatch, tmp_path):
    image_path = tmp_path / "a.jpg"
    _image(image_path)
    monkeypatch.setattr(gif, "get_sign_images", lambda: {"a": str(image_path)})

    response = await client.post("/api/text-to-gif", json={"text": "a"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"
    assert response.content.startswith(b"GIF")


async def test_text_to_gif_reports_missing_assets(client, monkeypatch):
    def unavailable():
        raise ServiceUnavailableError("sign images unavailable")

    monkeypatch.setattr(gif, "get_sign_images", unavailable)
    response = await client.post("/api/text-to-gif", json={"text": "hello"})
    assert response.status_code == 503
    assert response.json()["detail"] == "sign images unavailable"


async def test_text_to_gif_does_not_silently_skip_missing_letters(client, monkeypatch, tmp_path):
    image_path = tmp_path / "a.jpg"
    _image(image_path)
    monkeypatch.setattr(gif, "get_sign_images", lambda: {"a": str(image_path)})

    response = await client.post("/api/text-to-gif", json={"text": "ab"})
    assert response.status_code == 422
    assert "b" in response.json()["detail"]


async def test_speech_to_text_rejects_empty_upload(client):
    response = await client.post(
        "/api/speech-to-text",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 422


async def test_speech_to_text_rejects_wrong_media_type(client):
    response = await client.post(
        "/api/speech-to-text",
        files={"audio": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415


async def test_speech_to_text_enforces_upload_limit(client, monkeypatch):
    monkeypatch.setattr(voice, "get_settings", lambda: SimpleNamespace(max_upload_bytes=3))
    response = await client.post(
        "/api/speech-to-text",
        files={"audio": ("large.wav", b"1234", "audio/wav")},
    )
    assert response.status_code == 413


async def test_speech_to_text_uses_configured_service(client, monkeypatch):
    service = SimpleNamespace(transcribe=lambda _path: "hello world")
    monkeypatch.setattr(voice, "get_stt", lambda: service)
    response = await client.post(
        "/api/speech-to-text",
        files={"audio": ("speech.wav", b"audio bytes", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}


async def test_text_to_speech_returns_wav(client, monkeypatch):
    service = SimpleNamespace(synthesize=lambda _text: b"RIFF-test-wave")
    monkeypatch.setattr(voice, "get_tts", lambda: service)
    response = await client.post("/api/text-to-speech", json={"text": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


async def test_voice_to_gif_combines_transcription_and_rendering(client, monkeypatch, tmp_path):
    image_path = tmp_path / "a.jpg"
    _image(image_path)
    monkeypatch.setattr(voice, "get_stt", lambda: SimpleNamespace(transcribe=lambda _path: "a"))
    monkeypatch.setattr(voice, "get_sign_images", lambda: {"a": str(image_path)})

    response = await client.post(
        "/api/voice-to-gif",
        files={"audio": ("speech.wav", io.BytesIO(b"audio bytes"), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.content.startswith(b"GIF")
