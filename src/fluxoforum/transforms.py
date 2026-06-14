"""Pixel camera transforms and model-free optical flow."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def camera_transform(image: Image.Image, motion: dict[str, float]) -> Image.Image:
    width, height = image.size
    zoom = max(0.05, float(motion.get("zoom", 1.0)))
    angle = float(motion.get("angle", 0.0))
    tx = float(motion.get("translation_x", 0.0))
    ty = float(motion.get("translation_y", 0.0))
    px = float(motion.get("perspective_x", 0.0))
    py = float(motion.get("perspective_y", 0.0))

    angle_radians = math.radians(angle)
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    inverse = 1.0 / zoom
    center_x, center_y = width / 2.0, height / 2.0
    a = inverse * cosine
    b = inverse * sine
    d = -inverse * sine
    e = inverse * cosine
    c = center_x - a * center_x - b * center_y - tx
    f = center_y - d * center_x - e * center_y - ty
    coefficients = (a, b, c, d, e, f, px / max(width, 1), py / max(height, 1))
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=None,
    )


def optical_flow(previous: Image.Image, current: Image.Image, method: str = "dis") -> np.ndarray:
    import cv2

    previous_gray = cv2.cvtColor(np.asarray(previous.convert("RGB")), cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(np.asarray(current.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 0.5, 3, 21, 3, 7, 1.5, 0
        )
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setFinestScale(0)
    return estimator.calc(previous_gray, current_gray, None)


def flow_warp(image: Image.Image, flow: np.ndarray) -> Image.Image:
    import cv2

    pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        pixels, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return Image.fromarray(warped)

