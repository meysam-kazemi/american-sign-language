import { useState } from "react";
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

  return (
    <div className="app">
      <header>
        <h1>American Sign Language</h1>
        <p className="subtitle">Two-way ASL translation</p>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="panel">
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
