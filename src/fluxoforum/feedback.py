"""Pixel-space feedback and anti-drift corrections."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import FeedbackConfig


def _match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_mean, source_std = source.mean(), source.std()
    reference_mean, reference_std = reference.mean(), reference.std()
    return (source - source_mean) * (reference_std / (source_std + 1e-6)) + reference_mean


def color_match(image: Image.Image, reference: Image.Image, mode: str) -> Image.Image:
    if mode == "None":
        return image
    import cv2

    source = np.asarray(image.convert("RGB"))
    target = np.asarray(reference.convert("RGB").resize(image.size))
    if mode == "LAB":
        source_space = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_space = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        matched = np.stack(
            [_match_channel(source_space[..., i], target_space[..., i]) for i in range(3)],
            axis=-1,
        )
        matched = cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        source_float, target_float = source.astype(np.float32), target.astype(np.float32)
        matched = np.stack(
            [_match_channel(source_float[..., i], target_float[..., i]) for i in range(3)],
            axis=-1,
        )
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))


def edge_variance(image: Image.Image) -> float:
    import cv2

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_feedback(
    image: Image.Image,
    reference: Image.Image,
    previous: Image.Image | None,
    config: FeedbackConfig,
    noise: np.ndarray | None,
    noise_amount: float,
) -> Image.Image:
    result = color_match(image, reference, config.color_mode)
    contrast = config.contrast
    sharpen = config.sharpen
    if config.adaptive_correction and previous is not None:
        current_edges = edge_variance(result)
        previous_edges = edge_variance(previous)
        if current_edges < previous_edges * 0.85:
            sharpen = min(0.4, sharpen + (1.0 - current_edges / (previous_edges + 1e-6)) * 0.2)
        current_std = np.asarray(result).std()
        reference_std = np.asarray(reference).std()
        if current_std > reference_std * 1.15:
            contrast = min(contrast, 0.95)
    if contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if sharpen > 0:
        result = result.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=max(1, int(sharpen * 180)), threshold=2)
        )
    if noise is not None and noise_amount > 0:
        pixels = np.asarray(result, dtype=np.float32)
        pixels += noise * (noise_amount * 255.0)
        result = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return result

