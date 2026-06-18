import { useState } from "react";

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
        <p className="placeholder">Coming soon: {TABS.find((t) => t.id === tab).label}</p>
      </main>
    </div>
  );
}
