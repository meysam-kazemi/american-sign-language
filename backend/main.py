"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import gif, recognize, voice

app = FastAPI(title="American Sign Language API", version="0.1.0")

# Permissive CORS for local development (lock this down before deploying).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gif.router)
app.include_router(voice.router)
app.include_router(recognize.router)


@app.get("/health")
def health():
    return {"status": "ok"}
