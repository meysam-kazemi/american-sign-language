import { useEffect, useState } from "react";
import { getApplicationStatus } from "./api";
import CameraToSpeech from "./components/CameraToSpeech.jsx";
import LiveRecognition from "./components/LiveRecognition.jsx";
import TextToGif from "./components/TextToGif.jsx";
import VoiceToGif from "./components/VoiceToGif.jsx";

const TABS = [
  { id: "live", label: "Live Recognition" },
  { id: "text", label: "Text → GIF" },
  { id: "voice", label: "Voice → GIF" },
  { id: "speak", label: "Camera → Speech" },
];

export default function App() {
  const [tab, setTab] = useState("live");
  const [backendStatus, setBackendStatus] = useState("checking");

  useEffect(() => {
    let current = true;
    getApplicationStatus()
      .then((result) => current && setBackendStatus(result.status))
      .catch(() => current && setBackendStatus("offline"));
    return () => {
      current = false;
    };
  }, []);

  return (
    <div className="app">
      <header>
        <div className="header-row">
          <div>
            <h1>American Sign Language</h1>
            <p className="subtitle">Two-way ASL fingerspelling</p>
          </div>
          <span className={`health ${backendStatus}`} aria-live="polite">
            API: {backendStatus}
          </span>
        </div>
      </header>

      <nav className="tabs" aria-label="Translation modes" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
            aria-selected={tab === t.id}
            role="tab"
          >
            {t.label}
          </button>
        ))}
      </nav>

      {backendStatus === "degraded" && (
        <p className="notice setup-notice">
          Some ML assets are not ready. Check <code>/api/status</code> and the setup guide.
        </p>
      )}
      {backendStatus === "offline" && (
        <p className="error setup-notice" role="alert">
          The backend is offline. Start it before using a translation flow.
        </p>
      )}

      <main className="panel" role="tabpanel">
        {tab === "live" ? (
          <LiveRecognition />
        ) : tab === "text" ? (
          <TextToGif />
        ) : tab === "voice" ? (
          <VoiceToGif />
        ) : (
          <CameraToSpeech />
        )}
      </main>
    </div>
  );
}
