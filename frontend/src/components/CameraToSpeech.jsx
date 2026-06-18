import { useEffect, useRef, useState } from "react";
import { recognizeWsUrl, textToSpeech } from "../api";
import { useWebcam } from "../hooks/useWebcam";

const FRAME_INTERVAL_MS = 150;

export default function CameraToSpeech() {
  const { videoRef, active, error, start, stop } = useWebcam();
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const [text, setText] = useState("");
  const [status, setStatus] = useState("idle");
  const [audioUrl, setAudioUrl] = useState(null);
  const [speaking, setSpeaking] = useState(false);
  const [speakError, setSpeakError] = useState("");

  function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => blob && blob.arrayBuffer().then((b) => ws.send(b)), "image/jpeg", 0.7);
  }

  async function handleStart() {
    await start();
    const ws = new WebSocket(recognizeWsUrl());
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    setStatus("connecting");
    ws.onopen = () => {
      setStatus("running");
      timerRef.current = setInterval(captureFrame, FRAME_INTERVAL_MS);
    };
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.error) return setStatus("error: " + data.error);
      setText(data.text);
    };
    ws.onclose = () => setStatus("stopped");
    ws.onerror = () => setStatus("connection error");
  }

  function handleStop() {
    clearInterval(timerRef.current);
    wsRef.current?.close();
    wsRef.current = null;
    stop();
    setStatus("idle");
  }

  async function handleSpeak() {
    setSpeakError("");
    setSpeaking(true);
    try {
      setAudioUrl(await textToSpeech(text.trim()));
    } catch (err) {
      setSpeakError(err.message);
    } finally {
      setSpeaking(false);
    }
  }

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      wsRef.current?.close();
    };
  }, []);

  return (
    <div>
      <h2>Camera → Text &amp; Speech</h2>
      <video ref={videoRef} muted playsInline style={{ width: "100%", borderRadius: 8, background: "#000" }} />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        {!active ? (
          <button className="btn" onClick={handleStart}>Start camera</button>
        ) : (
          <button className="btn" onClick={handleStop}>Stop</button>
        )}
        <button className="btn" disabled={!text.trim() || speaking} onClick={handleSpeak}>
          {speaking ? "Synthesizing…" : "Speak text"}
        </button>
      </div>

      {(error || speakError) && <p className="error">{error || speakError}</p>}
      <p style={{ color: "#94a3b8", marginTop: "0.75rem" }}>Status: {status}</p>
      <div style={{ marginTop: "0.5rem", minHeight: "2rem", fontSize: "1.4rem" }}>
        {text || <span style={{ color: "#94a3b8" }}>Recognized text will appear here…</span>}
      </div>
      {audioUrl && <audio controls src={audioUrl} style={{ marginTop: "1rem", width: "100%" }} />}
    </div>
  );
}
