import { useCallback, useEffect, useRef, useState } from "react";
import { recognizeWsUrl } from "../api";
import { useWebcam } from "./useWebcam";

const FRAME_INTERVAL_MS = 150;
const MAX_BUFFERED_BYTES = 512 * 1024;

/** Share camera capture, WebSocket recognition, and cleanup between camera flows. */
export function useRecognition() {
  const webcam = useWebcam();
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const manualStopRef = useRef(false);
  const [text, setText] = useState("");
  const [label, setLabel] = useState(null);
  const [status, setStatus] = useState("idle");
  const [connectionError, setConnectionError] = useState("");

  const clearTimer = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
  }, []);

  const captureFrame = useCallback(() => {
    const video = webcam.videoRef.current;
    const ws = wsRef.current;
    if (!video || !ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
    if (ws.bufferedAmount > MAX_BUFFERED_BYTES) return;

    if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob && wsRef.current === ws && ws.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then((buffer) => {
          if (wsRef.current === ws && ws.readyState === WebSocket.OPEN) ws.send(buffer);
        });
      }
    }, "image/jpeg", 0.7);
  }, [webcam.videoRef]);

  const stop = useCallback(() => {
    manualStopRef.current = true;
    clearTimer();
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState < WebSocket.CLOSING) ws.close(1000, "Stopped by user");
    webcam.stop();
    setStatus("idle");
  }, [clearTimer, webcam.stop]);

  const start = useCallback(async () => {
    if (wsRef.current) return;
    setConnectionError("");
    setStatus("requesting-camera");
    const cameraStarted = await webcam.start();
    if (!cameraStarted) {
      setStatus("camera-error");
      return;
    }

    manualStopRef.current = false;
    setStatus("connecting");
    let ws;
    try {
      ws = new WebSocket(recognizeWsUrl());
    } catch (error) {
      webcam.stop();
      setConnectionError(error.message || "Could not create the recognition connection");
      setStatus("connection-error");
      return;
    }

    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      setStatus("running");
      timerRef.current = window.setInterval(captureFrame, FRAME_INTERVAL_MS);
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setConnectionError(data.error);
          setStatus(data.code === "service_unavailable" ? "unavailable" : "connection-error");
          return;
        }
        setLabel(data.label);
        setText(data.text);
      } catch {
        setConnectionError("The server returned an invalid recognition message");
        setStatus("connection-error");
      }
    };
    ws.onerror = () => {
      setConnectionError("The recognition connection failed");
      setStatus("connection-error");
    };
    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
      clearTimer();
      webcam.stop();
      setStatus(manualStopRef.current ? "idle" : "stopped");
    };
  }, [captureFrame, clearTimer, webcam.start, webcam.stop]);

  const reset = useCallback(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send("reset");
    setText("");
    setLabel(null);
  }, []);

  useEffect(
    () => () => {
      clearTimer();
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws && ws.readyState < WebSocket.CLOSING) ws.close(1000, "Component unmounted");
    },
    [clearTimer]
  );

  return {
    videoRef: webcam.videoRef,
    active: webcam.active,
    error: webcam.error || connectionError,
    text,
    label,
    status,
    start,
    stop,
    reset,
  };
}
