"""Flow 1: live ASL recognition over a WebSocket.

The client streams JPEG/PNG frames as binary messages. For each frame the server
detects hand landmarks, classifies the sign, feeds it through the StreamDecoder,
and replies with JSON {label, confidence, text}. A text message "reset" clears
the accumulated sentence.
"""
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services import get_classifier, get_detector
from ml.config import get_settings
from ml.decode import StreamDecoder

router = APIRouter(tags=["recognize"])


@router.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    await ws.accept()
    settings = get_settings()
    detector = get_detector()
    try:
        classifier = get_classifier()
    except Exception as exc:  # model missing / failed to load
        await ws.send_json({"error": str(getattr(exc, "detail", exc))})
        await ws.close()
        return

    decoder = StreamDecoder(
        stability=settings.decode_stability,
        min_confidence=settings.decode_min_confidence,
    )

    try:
        while True:
            message = await ws.receive()
            if "text" in message and message["text"] == "reset":
                decoder.reset()
                await ws.send_json({"label": None, "confidence": 0.0, "text": ""})
                continue
            if "bytes" not in message:
                continue

            frame = cv2.imdecode(np.frombuffer(message["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            features = detector.extract_features(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if features is None:
                text = decoder.update(None, 0.0)
                await ws.send_json({"label": None, "confidence": 0.0, "text": text})
                continue

            label, confidence = classifier.predict(features)
            text = decoder.update(label, confidence)
            await ws.send_json({"label": label, "confidence": confidence, "text": text})
    except WebSocketDisconnect:
        pass
