"""Build an animated ASL GIF from text by fingerspelling each character."""

import io
import os
import string

from PIL import Image

from ml.config import get_settings


class MissingSignImagesError(ValueError):
    """Raised when strict GIF generation cannot resolve all requested signs."""

    def __init__(self, signs):
        self.signs = tuple(dict.fromkeys(signs))
        super().__init__("Missing sign images for: " + ", ".join(self.signs))


def load_sign_images(base_dir=None):
    """Map each lowercase sign name to a representative image path.

    Looks for ``<sign>/<sign>1.jpg`` (the dataset naming convention) and falls
    back to the first image found in each sign's folder.
    """
    base_dir = base_dir or get_settings().signs_dir
    sign_images = {}
    for sign_name in os.listdir(base_dir):
        sign_dir = os.path.join(base_dir, sign_name)
        if not os.path.isdir(sign_dir):
            continue
        preferred = os.path.join(sign_dir, f"{sign_name}1.jpg")
        if os.path.exists(preferred):
            sign_images[sign_name.lower()] = preferred
        else:
            imgs = [
                filename
                for filename in sorted(os.listdir(sign_dir))
                if filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))
            ]
            if imgs:
                sign_images[sign_name.lower()] = os.path.join(sign_dir, imgs[0])
    return sign_images


def map_text_to_signs(text):
    """Map input text to a sequence of sign names (letters, digits, 'space')."""
    processed = text.lower().translate(str.maketrans("", "", string.punctuation))
    signs = []
    for ch in processed:
        if ch == " ":
            signs.append("space")
        elif ch in string.ascii_lowercase or ch in string.digits:
            signs.append(ch)
    return signs


def get_image_sequence(signs, sign_images):
    """Resolve sign names to image paths, skipping any that are missing."""
    sequence = []
    for name in signs:
        path = sign_images.get(name)
        if path:
            sequence.append(path)
    return sequence


def build_gif(
    text,
    sign_images=None,
    duration=300,
    loop=0,
    size=(200, 200),
    *,
    strict=False,
):
    """Return raw GIF bytes for the given text, or None if nothing maps."""
    sign_images = sign_images if sign_images is not None else load_sign_images()
    signs = map_text_to_signs(text)
    missing = [name for name in signs if name not in sign_images]
    if strict and missing:
        raise MissingSignImagesError(missing)
    paths = get_image_sequence(signs, sign_images)
    if not paths:
        return None

    frames = []
    for path in paths:
        try:
            with Image.open(path) as image:
                frames.append(image.convert("RGB").resize(size))
        except (FileNotFoundError, OSError):
            continue
    if not frames:
        return None

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:], duration=duration, loop=loop
    )
    return buf.getvalue()


def save_gif(text, output_path, **kwargs):
    """Build a GIF for `text` and write it to `output_path`."""
    data = build_gif(text, **kwargs)
    if data is None:
        return False
    with open(output_path, "wb") as f:
        f.write(data)
    return True


if __name__ == "__main__":
    ok = save_gif("hello world", "asl_animation.gif")
    print("Saved asl_animation.gif" if ok else "No signs mapped; nothing saved.")
