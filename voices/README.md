# Piper voice directory

Download both the Piper ONNX voice model and its sibling JSON configuration:

```bash
uv run python -m piper.download_voices en_US-amy-medium --download-dir voices
```

The default runtime paths are `voices/en_US-amy-medium.onnx` and
`voices/en_US-amy-medium.onnx.json`.
