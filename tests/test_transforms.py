import numpy as np
from PIL import Image

from fluxoforum.feedback import color_match
from fluxoforum.noise import CoherentNoise
from fluxoforum.transforms import camera_transform


def test_camera_transform_keeps_size():
    image = Image.new("RGB", (256, 256), "red")
    result = camera_transform(image, {"zoom": 1.05, "angle": 2, "translation_x": 3})
    assert result.size == image.size


def test_noise_is_deterministic():
    first = CoherentNoise(42)
    second = CoherentNoise(42)
    motion = {"zoom": 1.0}
    assert np.allclose(first.sample(0, (32, 32), motion), second.sample(0, (32, 32), motion))


def test_color_match_keeps_image_contract():
    image = Image.new("RGB", (32, 32), (20, 40, 60))
    reference = Image.new("RGB", (32, 32), (100, 120, 140))
    result = color_match(image, reference, "RGB")
    assert result.mode == "RGB"
    assert result.size == image.size

