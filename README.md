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

## Project layout

```
ml/   importable ML package: landmarks, classifier, training, gif builder, stream decoder
```
