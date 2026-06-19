import { useState } from "react";
import { textToGif } from "../api";

export default function TextToGif() {
  const [text, setText] = useState("");
  const [gifUrl, setGifUrl] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleGenerate() {
    setError("");
    setLoading(true);
    try {
      const url = await textToGif(text.trim());
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
      <h2>Text → ASL GIF</h2>
      <textarea
        rows={3}
        value={text}
        placeholder="Type a word or sentence…"
        onChange={(e) => setText(e.target.value)}
        style={{ width: "100%", padding: "0.6rem", borderRadius: 8 }}
      />
      <div style={{ marginTop: "0.75rem" }}>
        <button className="btn" disabled={!text.trim() || loading} onClick={handleGenerate}>
          {loading ? "Generating…" : "Generate GIF"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {gifUrl && (
        <div style={{ marginTop: "1rem" }}>
          <img src={gifUrl} alt="ASL animation" style={{ maxWidth: "100%", borderRadius: 8 }} />
        </div>
      )}
    </div>
  );
}
