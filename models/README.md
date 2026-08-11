# Trained model directory

Running `uv run python -m ml.train` writes:

- `asl_landmark_mlp.pth`
- `labels.json`

Both runtime artifacts are ignored by Git and must be mounted or generated for
live recognition.
