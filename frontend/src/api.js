// Backend endpoints. Base URLs come from Vite env (see .env.example).
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

export const recognizeWsUrl = () => `${WS_URL}/ws/recognize`;

// Flow 2: text -> ASL GIF. Returns an object URL for an <img>.
export async function textToGif(text) {
  const res = await fetch(`${API_URL}/api/text-to-gif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return URL.createObjectURL(await res.blob());
}

// Flow 3: spoken audio -> ASL GIF. Returns an object URL for an <img>.
export async function voiceToGif(audioBlob) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  const res = await fetch(`${API_URL}/api/voice-to-gif`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorText(res));
  return URL.createObjectURL(await res.blob());
}

// Flow 4: text -> speech (WAV). Returns an object URL for an <audio>.
export async function textToSpeech(text) {
  const res = await fetch(`${API_URL}/api/text-to-speech`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return URL.createObjectURL(await res.blob());
}

async function errorText(res) {
  try {
    const data = await res.json();
    return data.detail || `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}
