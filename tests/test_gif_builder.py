import io

import pytest
from PIL import Image

from ml.gif_builder import (
    MissingSignImagesError,
    build_gif,
    get_image_sequence,
    load_sign_images,
    map_text_to_signs,
)


def test_map_text_lowercases_and_strips_punctuation():
    assert map_text_to_signs("Hi!") == ["h", "i"]


def test_map_text_spaces_become_space_token():
    assert map_text_to_signs("a b") == ["a", "space", "b"]


def test_map_text_keeps_digits():
    assert map_text_to_signs("a1") == ["a", "1"]


def test_map_text_ignores_unsupported_unicode():
    assert map_text_to_signs("aé") == ["a"]


def test_get_image_sequence_skips_missing():
    images = {"a": "/path/a.jpg"}
    assert get_image_sequence(["a", "b"], images) == ["/path/a.jpg"]


def _write_image(path):
    Image.new("RGB", (10, 10), "white").save(path)


def test_build_gif_returns_bytes(tmp_path):
    a = tmp_path / "a.jpg"
    _write_image(a)
    data = build_gif("aa", sign_images={"a": str(a)})
    assert data is not None
    img = Image.open(io.BytesIO(data))
    assert img.format == "GIF"


def test_build_gif_returns_none_when_nothing_maps():
    assert build_gif("xyz", sign_images={}) is None


def test_strict_gif_reports_every_missing_sign(tmp_path):
    a = tmp_path / "a.jpg"
    _write_image(a)
    with pytest.raises(MissingSignImagesError) as exc_info:
        build_gif("abc", sign_images={"a": str(a)}, strict=True)
    assert exc_info.value.signs == ("b", "c")


def test_load_sign_images_uses_preferred_and_fallback_files(tmp_path):
    a_dir = tmp_path / "A"
    b_dir = tmp_path / "B"
    a_dir.mkdir()
    b_dir.mkdir()
    _write_image(a_dir / "A1.jpg")
    _write_image(b_dir / "example.png")

    images = load_sign_images(tmp_path)
    assert images == {"a": str(a_dir / "A1.jpg"), "b": str(b_dir / "example.png")}
