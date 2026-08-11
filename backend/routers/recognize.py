"""Flow 1: live ASL recognition over a WebSocket.

The client streams JPEG/PNG frames as binary messages. For each frame the server
detects hand landmarks, classifies the sign, feeds it through the StreamDecoder,
and replies with JSON {label, confidence, text}. A text message "reset" clears
the accumulated sentence.
"""

import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services import (
    get_classifier,
    get_detector,
    predict_sign,
)
from ml.config import get_settings
from ml.decode import StreamDecoder

router = APIRouter(tags=["recognize"])


@router.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    await ws.accept()
    settings = get_settings()
    try:
        get_classifier()
        get_detector()
    except Exception as exc:
        await ws.send_json({"error": str(exc), "code": "service_unavailable"})
        await ws.close(code=1013)
        return

    decoder = StreamDecoder(
        stability=settings.decode_stability,
        min_confidence=settings.decode_min_confidence,
    )

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "text" in message:
                if message["text"] == "reset":
                    decoder.reset()
                    await ws.send_json({"label": None, "confidence": 0.0, "text": ""})
                else:
                    await ws.send_json({"error": "Unsupported command.", "code": "bad_command"})
                continue
            if "bytes" not in message:
                continue
            if len(message["bytes"]) > settings.max_frame_bytes:
                await ws.send_json({"error": "Frame is too large.", "code": "frame_too_large"})
                continue

            frame = cv2.imdecode(np.frombuffer(message["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json(
                    {"error": "Frame could not be decoded.", "code": "invalid_frame"}
                )
                continue

            prediction = await asyncio.to_thread(predict_sign, frame)
            if prediction is None:
                text = decoder.update(None, 0.0)
                await ws.send_json({"label": None, "confidence": 0.0, "text": text})
                continue

            label, confidence = prediction
            text = decoder.update(label, confidence)
            await ws.send_json({"label": label, "confidence": confidence, "text": text})
    except WebSocketDisconnect:
        pass
