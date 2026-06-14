from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


class FakePipeline:
    loader_name = "fake"

    def __init__(self):
        self.calls = []

    def generate(
        self,
        prompt,
        image,
        width,
        height,
        steps,
        guidance,
        seed,
        strength,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "has_image": image is not None,
                "seed": seed,
                "strength": strength,
            }
        )
        rng = np.random.default_rng(seed)
        color = tuple(int(value) for value in rng.integers(40, 220, size=3))
        generated = Image.new("RGB", (width, height), color)
        if image is not None:
            generated = Image.blend(image.resize((width, height)), generated, strength)
        return generated

    def memory_stats(self):
        return {"peak_reserved_gb": 1.0}


@pytest.fixture
def fake_pipeline():
    return FakePipeline()

