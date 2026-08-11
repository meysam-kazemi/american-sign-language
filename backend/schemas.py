"""Request and response models for the API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class TextResponse(BaseModel):
    text: str


class RecognitionResult(BaseModel):
    label: str | None = None
    confidence: float = 0.0
    text: str = ""


class CapabilityStatus(BaseModel):
    available: bool
    detail: str


class ApplicationStatus(BaseModel):
    status: Literal["ready", "degraded"]
    capabilities: dict[str, CapabilityStatus]
