import io

from PIL import Image

from ml.gif_builder import build_gif, get_image_sequence, map_text_to_signs


def test_map_text_lowercases_and_strips_punctuation():
    assert map_text_to_signs("Hi!") == ["h", "i"]


def test_map_text_spaces_become_space_token():
    assert map_text_to_signs("a b") == ["a", "space"]


def test_map_text_keeps_digits():
    assert map_text_to_signs("a1") == ["a", "1"]


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
