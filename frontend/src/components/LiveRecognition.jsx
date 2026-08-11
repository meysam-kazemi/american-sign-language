import { useRecognition } from "../hooks/useRecognition";

export default function LiveRecognition() {
  const recognition = useRecognition();
  const { videoRef, active, error, text, label, status, start, stop, reset } = recognition;

  return (
    <div>
      <h2>Live ASL Recognition</h2>
      <p className="help">Show one alphabet sign at a time and briefly lower your hand between repeated letters.</p>
      <video ref={videoRef} muted playsInline className="camera" aria-label="Camera preview" />

      <div className="actions">
        {!active ? (
          <button className="btn" onClick={start}>Start camera</button>
        ) : (
          <button className="btn secondary" onClick={stop}>Stop</button>
        )}
        <button className="btn secondary" disabled={!active} onClick={reset}>Reset text</button>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      <p className="status" aria-live="polite">
        Status: {status}{label ? ` · current sign: ${label}` : ""}
      </p>
      <div className="result-text" aria-live="polite">
        {text || <span className="muted">Recognized text will appear here…</span>}
      </div>
    </div>
  );
}
