// Empty defaults use the current origin in production. Vite's dev server proxies
// these paths to the local backend (see vite.config.js).
const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

function defaultWsUrl() {
  if (typeof window === "undefined") return "ws://localhost:8000";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

const WS_URL = (import.meta.env.VITE_WS_URL || defaultWsUrl()).replace(/\/$/, "");

export const recognizeWsUrl = () => `${WS_URL}/ws/recognize`;

export async function getApplicationStatus() {
  const res = await fetch(`${API_URL}/api/status`);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

// Flow 2: text -> ASL GIF. Returns a Blob owned by the caller.
export async function textToGif(text) {
  const res = await fetch(`${API_URL}/api/text-to-gif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.blob();
}

// Flow 3: spoken audio -> ASL GIF. Returns a Blob owned by the caller.
export async function voiceToGif(audioBlob) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  const res = await fetch(`${API_URL}/api/voice-to-gif`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorText(res));
  return res.blob();
}

// Flow 4: text -> speech (WAV). Returns a Blob owned by the caller.
export async function textToSpeech(text) {
  const res = await fetch(`${API_URL}/api/text-to-speech`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.blob();
}

export async function errorText(res) {
  try {
    const data = await res.json();
    return data.detail || `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}
