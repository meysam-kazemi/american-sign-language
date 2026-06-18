import { useState } from "react";
import { voiceToGif } from "../api";
import { useRecorder } from "../hooks/useRecorder";

export default function VoiceToGif() {
  const { recording, error: micError, start, stop } = useRecorder();
  const [gifUrl, setGifUrl] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleStop() {
    const blob = await stop();
    if (!blob) return;
    setError("");
    setLoading(true);
    try {
      const url = await voiceToGif(blob);
      setGifUrl(url);
    } catch (err) {
      setError(err.message);
      setGifUrl(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Voice → ASL GIF</h2>
      <p style={{ color: "#94a3b8" }}>Record a sentence; it gets transcribed and turned into signs.</p>
      <div style={{ display: "flex", gap: "0.5rem" }}>
        {!recording ? (
          <button className="btn" disabled={loading} onClick={start}>
            {loading ? "Processing…" : "Start recording"}
          </button>
        ) : (
          <button className="btn" onClick={handleStop}>Stop &amp; generate</button>
        )}
      </div>
      {(error || micError) && <p className="error">{error || micError}</p>}
      {gifUrl && (
        <div style={{ marginTop: "1rem" }}>
          <img src={gifUrl} alt="ASL animation" style={{ maxWidth: "100%", borderRadius: 8 }} />
        </div>
      )}
    </div>
  );
}
