"""Flow 2: text -> ASL GIF."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.schemas import TextRequest
from backend.services import get_sign_images
from ml.gif_builder import build_gif

router = APIRouter(prefix="/api", tags=["gif"])


@router.post("/text-to-gif")
def text_to_gif(req: TextRequest):
    """Return an animated ASL GIF (image/gif) for the given text."""
    data = build_gif(req.text, sign_images=get_sign_images())
    if data is None:
        raise HTTPException(status_code=422, detail="No signs could be mapped from the text.")
    return Response(content=data, media_type="image/gif")
