# American Sign Language fingerspelling

A local-first, two-way accessibility MVP for recognizing ASL alphabet signs and
rendering English text as animated fingerspelling. It includes a FastAPI backend,
a React frontend, MediaPipe landmark extraction, a trainable PyTorch classifier,
Whisper speech recognition, and Piper text-to-speech.

> This project performs alphabet fingerspelling. It is not a full ASL language
> translator and does not model ASL grammar, facial expression, or continuous
> signing. Dynamic letters such as J and Z need motion-aware training beyond the
> included static landmark classifier.

## Capabilities and required assets

| Flow | Runtime | Required local assets |
| --- | --- | --- |
| Camera → text | MediaPipe + trained MLP | `models/asl_landmark_mlp.pth`, `models/labels.json` |
| Text → GIF | Pillow | Representative images below `ASL_SIGNS_DIR` |
| Voice → GIF | Whisper + Pillow | Whisper model cache and representative sign images |
| Camera → speech | Recognition + Piper | Classifier artifacts and Piper `.onnx`/`.onnx.json` files |

The application starts without these large or licensed artifacts. `GET /api/status`
reports exactly which capabilities are ready, and unavailable endpoints return a
descriptive `503` instead of crashing.

## Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 22 and npm 10 for local frontend development
- A camera and/or microphone for media flows
- Docker with Compose v2 for containerized startup (optional)

Linux and Windows resolve the CPU-only PyTorch wheel by default, avoiding a
multi-gigabyte CUDA installation. Change the uv Torch source if GPU training is
required.

## Local setup

Install the locked backend and frontend dependencies:

```bash
make setup
cp .env.example .env  # optional; the checked-in defaults work from the repo root
```

### 1. Add the ASL image dataset

Place an ASL Alphabet-style dataset below `dataset/asl_alphabet_train/`:

```text
dataset/asl_alphabet_train/
├── A/A1.jpg ...
├── B/B1.jpg ...
├── ...
├── space/space1.jpg ...
├── del/del1.jpg ...
└── nothing/nothing1.jpg ...
```

The raw dataset remains ignored by Git. See `dataset/README.md`. The GIF builder
uses one representative image from each class by default; a smaller production
directory can be selected with `ASL_SIGNS_DIR`.

### 2. Train the camera classifier

```bash
uv run python -m ml.preprocess
uv run python -m ml.train
```

This creates the ignored runtime artifacts in `models/`. Training uses a
deterministic seed and holds out 15% of the cached landmarks for validation.

### 3. Install the Piper voice

```bash
uv run python -m piper.download_voices en_US-amy-medium --download-dir voices
```

Both `voices/en_US-amy-medium.onnx` and
`voices/en_US-amy-medium.onnx.json` are required. Set `TTS_MODEL_PATH` to use a
different voice. If Piper is unavailable, the camera-to-speech page falls back
to the browser's speech synthesizer when supported.

### 4. Check readiness and run

```bash
make assets
```

`make assets` exits nonzero until every optional capability is ready, while
printing a JSON explanation for each one.

Start the backend and frontend in separate terminals:

```bash
make run
```

```bash
npm --prefix frontend run dev
```

Open <http://localhost:5173>. The Vite server proxies `/api`, `/health`, and
`/ws` to the backend at port 8000.

The first voice request downloads the configured Whisper model unless
`WHISPER_MODEL` points to an existing local model directory.

## Docker Compose

After populating `dataset/`, `models/`, and `voices/` on the host:

```bash
docker compose up --build
```

Open <http://localhost:5173>. The nginx frontend proxies API and WebSocket
traffic internally, the backend is health-gated, and the Whisper download cache
is persisted in a named volume. `.env` is optional; Compose uses safe defaults.

## API

- `GET /health` — process liveness and application version
- `GET /api/status` — per-capability asset readiness
- `POST /api/text-to-gif` — validated text to animated fingerspelling GIF
- `POST /api/voice-to-gif` — uploaded audio to transcript to GIF
- `POST /api/speech-to-text` — uploaded audio to text
- `POST /api/text-to-speech` — validated text to Piper WAV
- `WS /ws/recognize` — JPEG/PNG camera frames to live recognition results

Text requests are trimmed and limited to 500 characters. Audio uploads are
streamed to temporary storage, restricted to common audio media types, and
limited to 25 MiB by default. WebSocket frames default to a 2 MiB maximum. All
limits and allowed browser origins are configurable in `.env.example`.

Interactive API documentation is available at <http://localhost:8000/docs>.

## Verification

Run the complete local quality gate:

```bash
make check
docker compose config --quiet
npm --prefix frontend audit --audit-level=high
```

`make check` runs Ruff lint/format checks, 35 Python tests, 8 frontend tests, and
the production frontend build. GitHub Actions repeats these checks for pushes
and pull requests, and Dependabot monitors uv, npm, and workflow dependencies.

## Project layout

```text
backend/   FastAPI application, routers, validation, and lazy model services
frontend/  React/Vite interface, media hooks, nginx config, and component tests
ml/        Landmark preprocessing, classifier training/inference, decoding,
           GIF creation, Whisper STT, and Piper TTS
scripts/   Readiness utilities
tests/     ML behavior, HTTP API, upload limits, and WebSocket tests
```
