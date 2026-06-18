# american-sign-language

A two-way communication tool for American Sign Language (ASL): transcribe ASL
from a camera/video into text, and generate animated ASL GIFs from text or voice.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                 # install core ML dependencies
uv sync --extra backend # also install the API/voice stack (later phases)
cp .env.example .env     # configure paths and settings
```

## Configuration

All settings live in `.env` (see `.env.example`): dataset/model paths, MediaPipe
detection params, training hyperparameters, stream-decoding, and backend options.

## Training the ASL→text model

Dataset layout (one folder per sign class, e.g. the Kaggle ASL Alphabet set):

```
dataset/asl_alphabet_train/
├── A/  A1.jpg ...
├── B/
├── space/
└── ...
```

```bash
uv run python -m ml.preprocess   # extract hand landmarks -> dataset/landmarks/*.pkl (run once)
uv run python -m ml.train        # train classifier -> models/asl_landmark_mlp.pth + labels.json
```

## Running the backend (API)

```bash
uv sync --extra backend
uv run uvicorn backend.main:app --reload
```

Endpoints:
- `GET  /health` — liveness check
- `POST /api/text-to-gif` — Flow 2: text → ASL GIF
- `POST /api/voice-to-gif` — Flow 3: spoken audio → text → ASL GIF
- `POST /api/speech-to-text` — Whisper transcription
- `POST /api/text-to-speech` — Piper speech synthesis (Flow 4)
- `WS   /ws/recognize` — Flow 1: stream camera frames → live caption

Recognition endpoints require a trained model in `models/` (see Training).

## Running the frontend (React)

```bash
cd frontend
npm install
cp .env.example .env   # point VITE_API_URL / VITE_WS_URL at the backend
npm run dev            # dev server on http://localhost:5173
```

Tabs map to the four flows:
- **Live Recognition** (Flow 1) — webcam frames streamed over the WebSocket → live caption
- **Text → GIF** (Flow 2)
- **Voice → GIF** (Flow 3) — record audio, transcribe, render signs
- **Camera → Speech** (Flow 4) — recognize signs to text, then synthesize speech

## Project layout

```
ml/        importable ML package: landmarks, classifier, training, gif builder,
           stream decoder, speech-to-text, text-to-speech
backend/   FastAPI app: routers (gif, voice, recognize), lazy ML services
frontend/  Vite + React app: tabbed UI for the four flows
```
