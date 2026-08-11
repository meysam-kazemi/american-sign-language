"""Flow 2: text -> ASL GIF."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.schemas import TextRequest
from backend.services import ServiceUnavailableError, get_sign_images
from ml.gif_builder import MissingSignImagesError, build_gif

router = APIRouter(prefix="/api", tags=["gif"])


@router.post("/text-to-gif")
async def text_to_gif(req: TextRequest):
    """Return an animated ASL GIF (image/gif) for the given text."""
    try:
        data = build_gif(req.text, sign_images=get_sign_images(), strict=True)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MissingSignImagesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if data is None:
        raise HTTPException(status_code=422, detail="No signs could be mapped from the text.")
    return Response(
        content=data,
        media_type="image/gif",
        headers={"Cache-Control": "no-store"},
    )
