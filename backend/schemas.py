"""Request/response models for the API."""
from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    text: str


class RecognitionResult(BaseModel):
    label: str | None = None
    confidence: float = 0.0
    text: str = ""
