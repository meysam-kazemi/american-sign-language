import { useState } from "react";
import { textToGif } from "../api";
import { useObjectUrl } from "../hooks/useObjectUrl";

export default function TextToGif() {
  const [text, setText] = useState("");
  const gif = useObjectUrl();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleGenerate() {
    setError("");
    setLoading(true);
    try {
      gif.setBlob(await textToGif(text.trim()));
    } catch (err) {
      setError(err.message);
      gif.clear();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Text → ASL GIF</h2>
      <p className="help">Text is fingerspelled one character at a time using your configured sign images.</p>
      <label htmlFor="sign-text">Text to fingerspell</label>
      <textarea
        id="sign-text"
        rows={3}
        value={text}
        placeholder="Type a word or sentence…"
        onChange={(e) => setText(e.target.value)}
        maxLength={500}
      />
      <div className="actions">
        <button className="btn" disabled={!text.trim() || loading} onClick={handleGenerate}>
          {loading ? "Generating…" : "Generate GIF"}
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {gif.url && (
        <div className="media-result">
          <img src={gif.url} alt={`ASL fingerspelling for ${text.trim()}`} />
        </div>
      )}
    </div>
  );
}
