import { useState } from "react";
import { voiceToGif } from "../api";
import { useObjectUrl } from "../hooks/useObjectUrl";
import { useRecorder } from "../hooks/useRecorder";

export default function VoiceToGif() {
  const { recording, error: micError, start, stop } = useRecorder();
  const gif = useObjectUrl();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleStop() {
    const blob = await stop();
    if (!blob) return;
    setError("");
    setLoading(true);
    try {
      gif.setBlob(await voiceToGif(blob));
    } catch (err) {
      setError(err.message);
      gif.clear();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Voice → ASL GIF</h2>
      <p className="help">Record a short English sentence; it is transcribed and fingerspelled.</p>
      <div className="actions">
        {!recording ? (
          <button className="btn" disabled={loading} onClick={start}>
            {loading ? "Processing…" : "Start recording"}
          </button>
        ) : (
          <button className="btn" onClick={handleStop}>Stop &amp; generate</button>
        )}
      </div>
      {(error || micError) && <p className="error" role="alert">{error || micError}</p>}
      {gif.url && (
        <div className="media-result">
          <img src={gif.url} alt="ASL fingerspelling generated from speech" />
        </div>
      )}
    </div>
  );
}
