"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import gif, recognize, voice
from backend.schemas import ApplicationStatus
from backend.services import capability_status, clear_service_caches
from ml.config import get_settings


@asynccontextmanager
async def lifespan(_app):
    yield
    clear_service_caches()


settings = get_settings()
app = FastAPI(title="American Sign Language API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gif.router)
app.include_router(voice.router)
app.include_router(recognize.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/status", response_model=ApplicationStatus)
async def status():
    capabilities = capability_status()
    return {
        "status": "ready"
        if all(item["available"] for item in capabilities.values())
        else "degraded",
        "capabilities": capabilities,
    }
