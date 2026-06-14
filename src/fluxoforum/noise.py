"""Deterministic coherent pixel-noise state."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .transforms import camera_transform


@dataclass
class CoherentNoise:
    seed: int
    warp_blend: float = 0.8
    previous: np.ndarray | None = None

    def sample(
        self,
        frame: int,
        size: tuple[int, int],
        motion: dict[str, float],
    ) -> np.ndarray:
        width, height = size
        fresh = np.random.default_rng(self.seed + frame).normal(
            0.0, 1.0, (height, width, 3)
        ).astype(np.float32)
        if self.previous is None:
            result = fresh
        else:
            previous_image = Image.fromarray(self._to_uint8(self.previous))
            warped = np.asarray(camera_transform(previous_image, motion), dtype=np.float32)
            warped = (warped - 127.5) / 42.5
            result = self.warp_blend * warped + (1.0 - self.warp_blend) * fresh
        result -= result.mean(axis=(0, 1), keepdims=True)
        result /= result.std(axis=(0, 1), keepdims=True) + 1e-6
        self.previous = result
        return result

    @staticmethod
    def _to_uint8(noise: np.ndarray) -> np.ndarray:
        return np.clip(noise * 42.5 + 127.5, 0, 255).astype(np.uint8)

    def serialize(self) -> str | None:
        if self.previous is None:
            return None
        buffer = io.BytesIO()
        np.savez_compressed(buffer, previous=self.previous.astype(np.float16))
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def restore(self, value: str | None) -> None:
        if not value:
            self.previous = None
            return
        buffer = io.BytesIO(base64.b64decode(value))
        self.previous = np.load(buffer)["previous"].astype(np.float32)

