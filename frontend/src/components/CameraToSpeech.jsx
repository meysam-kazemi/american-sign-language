import { useState } from "react";
import { textToSpeech } from "../api";
import { useObjectUrl } from "../hooks/useObjectUrl";
import { useRecognition } from "../hooks/useRecognition";

export default function CameraToSpeech() {
  const { videoRef, active, error, text, status, start, stop } = useRecognition();
  const audio = useObjectUrl();
  const [speaking, setSpeaking] = useState(false);
  const [speakError, setSpeakError] = useState("");

  async function handleSpeak() {
    setSpeakError("");
    setSpeaking(true);
    try {
      audio.setBlob(await textToSpeech(text.trim()));
    } catch (err) {
      if ("speechSynthesis" in window && "SpeechSynthesisUtterance" in window) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(text.trim()));
        setSpeakError("Server voice unavailable; using this browser's voice instead.");
      } else {
        setSpeakError(err.message);
      }
    } finally {
      setSpeaking(false);
    }
  }

  return (
    <div>
      <h2>Camera → Text &amp; Speech</h2>
      <p className="help">Recognize a fingerspelled message, then play it as speech.</p>
      <video ref={videoRef} muted playsInline className="camera" aria-label="Camera preview" />

      <div className="actions">
        {!active ? (
          <button className="btn" onClick={start}>Start camera</button>
        ) : (
          <button className="btn secondary" onClick={stop}>Stop</button>
        )}
        <button className="btn" disabled={!text.trim() || speaking} onClick={handleSpeak}>
          {speaking ? "Synthesizing…" : "Speak text"}
        </button>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      {speakError && <p className="notice" role="status">{speakError}</p>}
      <p className="status" aria-live="polite">Status: {status}</p>
      <div className="result-text" aria-live="polite">
        {text || <span className="muted">Recognized text will appear here…</span>}
      </div>
      {audio.url && <audio controls autoPlay src={audio.url} className="audio" />}
    </div>
  );
}
