from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.routers import recognize
from backend.services import ServiceUnavailableError

pytestmark = pytest.mark.anyio


class FakeWebSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.sent = []
        self.accepted = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code=1000):
        self.close_code = code


def _jpeg_bytes():
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


async def test_websocket_recognizes_stable_predictions(monkeypatch):
    settings = SimpleNamespace(decode_stability=2, decode_min_confidence=0.6, max_frame_bytes=1024)
    monkeypatch.setattr(recognize, "get_settings", lambda: settings)
    monkeypatch.setattr(recognize, "get_detector", lambda: object())
    monkeypatch.setattr(recognize, "get_classifier", lambda: object())
    monkeypatch.setattr(recognize, "predict_sign", lambda _frame: ("a", 0.99))

    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": _jpeg_bytes()},
            {"type": "websocket.receive", "bytes": _jpeg_bytes()},
            {"type": "websocket.receive", "text": "reset"},
        ]
    )
    await recognize.recognize(websocket)

    assert websocket.accepted
    assert [message["text"] for message in websocket.sent] == ["", "A", ""]


async def test_websocket_rejects_invalid_frames(monkeypatch):
    settings = SimpleNamespace(decode_stability=2, decode_min_confidence=0.6, max_frame_bytes=1024)
    monkeypatch.setattr(recognize, "get_settings", lambda: settings)
    monkeypatch.setattr(recognize, "get_detector", lambda: object())
    monkeypatch.setattr(recognize, "get_classifier", lambda: object())

    websocket = FakeWebSocket([{"type": "websocket.receive", "bytes": b"not an image"}])
    await recognize.recognize(websocket)
    assert websocket.sent[0]["code"] == "invalid_frame"


async def test_websocket_reports_missing_classifier(monkeypatch):
    monkeypatch.setattr(recognize, "get_detector", lambda: object())

    def unavailable():
        raise ServiceUnavailableError("classifier unavailable")

    monkeypatch.setattr(recognize, "get_classifier", unavailable)
    websocket = FakeWebSocket()
    await recognize.recognize(websocket)
    assert websocket.sent == [{"error": "classifier unavailable", "code": "service_unavailable"}]
    assert websocket.close_code == 1013
