"""Safe Deforum-style schedule parsing and rendering."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


SCHEDULE_RE = re.compile(r"^\s*(\d+)\s*:\s*\((.*)\)\s*$")
ALLOWED_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "floor": math.floor,
    "ceil": math.ceil,
    "clamp": lambda value, low, high: max(low, min(high, value)),
    "lerp": lambda start, end, amount: start + (end - start) * amount,
}
ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


class ScheduleError(ValueError):
    pass


class SafeExpression:
    """Compile a numeric expression after validating its AST."""

    _allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    def __init__(self, expression: str):
        self.expression = expression.strip()
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise ScheduleError(f"invalid expression: {self.expression}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, self._allowed_nodes):
                raise ScheduleError(f"unsupported expression element: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                    raise ScheduleError("only approved numeric functions may be called")
                if node.keywords:
                    raise ScheduleError("keyword arguments are not supported")
            if isinstance(node, ast.Name):
                valid_names = {
                    "f",
                    "t",
                    "fps",
                    "total",
                    "beat",
                    "onset",
                    "rms",
                    "bass",
                    "mids",
                    "highs",
                    *ALLOWED_FUNCTIONS,
                    *ALLOWED_CONSTANTS,
                }
                if node.id not in valid_names:
                    raise ScheduleError(f"unknown expression name: {node.id}")
        self._code = compile(tree, "<schedule>", "eval")

    def evaluate(self, context: Mapping[str, float]) -> float:
        namespace = {**ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS, **context}
        try:
            value = eval(self._code, {"__builtins__": {}}, namespace)
            value = float(value)
        except Exception as exc:
            raise ScheduleError(f"failed to evaluate '{self.expression}': {exc}") from exc
        if not math.isfinite(value):
            raise ScheduleError(f"expression produced non-finite value: {self.expression}")
        return value


def split_schedule(text: str) -> list[str]:
    """Split comma-separated keyframes while preserving nested function commas."""
    chunks: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ScheduleError("unbalanced parentheses")
        elif char == "," and depth == 0:
            chunks.append(text[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ScheduleError("unbalanced parentheses")
    chunks.append(text[start:].strip())
    return [chunk for chunk in chunks if chunk]


def parse_schedule(text: str, total_frames: int) -> list[tuple[int, SafeExpression]]:
    if not text.strip():
        raise ScheduleError("schedule cannot be empty")
    keyframes: list[tuple[int, SafeExpression]] = []
    for chunk in split_schedule(text):
        match = SCHEDULE_RE.match(chunk)
        if not match:
            raise ScheduleError(f"expected frame:(expression), got: {chunk}")
        frame = int(match.group(1))
        if frame < 0 or frame >= total_frames:
            raise ScheduleError(f"frame {frame} is outside 0..{total_frames - 1}")
        keyframes.append((frame, SafeExpression(match.group(2))))
    keyframes.sort(key=lambda item: item[0])
    if len({frame for frame, _ in keyframes}) != len(keyframes):
        raise ScheduleError("duplicate keyframe")
    return keyframes


def ease(value: float, mode: str) -> float:
    value = max(0.0, min(1.0, value))
    if mode == "ease_in":
        return value * value
    if mode == "ease_out":
        return 1.0 - (1.0 - value) ** 2
    if mode == "ease_in_out":
        return value * value * (3.0 - 2.0 * value)
    if mode == "cosine":
        return (1.0 - math.cos(math.pi * value)) / 2.0
    return value


def render_schedule(
    text: str,
    total_frames: int,
    fps: float,
    audio: Mapping[str, np.ndarray] | None = None,
    interpolation: str = "linear",
) -> np.ndarray:
    keyframes = parse_schedule(text, total_frames)
    audio = audio or {}

    def context(frame: int) -> dict[str, float]:
        values = {"f": frame, "t": frame / fps, "fps": fps, "total": total_frames}
        for feature in ("beat", "onset", "rms", "bass", "mids", "highs"):
            array = audio.get(feature)
            values[feature] = float(array[frame]) if array is not None and frame < len(array) else 0.0
        return values

    if len(keyframes) == 1:
        _, expression = keyframes[0]
        return np.asarray(
            [expression.evaluate(context(frame)) for frame in range(total_frames)],
            dtype=np.float64,
        )

    evaluated = [(frame, expression.evaluate(context(frame))) for frame, expression in keyframes]
    result = np.empty(total_frames, dtype=np.float64)
    result[: evaluated[0][0]] = evaluated[0][1]
    result[evaluated[-1][0] :] = evaluated[-1][1]
    for (start_frame, start_value), (end_frame, end_value) in zip(evaluated, evaluated[1:]):
        span = end_frame - start_frame
        for frame in range(start_frame, end_frame):
            amount = ease((frame - start_frame) / span, interpolation)
            result[frame] = start_value + (end_value - start_value) * amount
    return result


def parse_prompt_schedule(text: str, default_prompt: str, total_frames: int) -> list[str]:
    """Parse one `frame: prompt` entry per line and hold prompts between keyframes."""
    entries = [(0, default_prompt)]
    for line in text.splitlines():
        if not line.strip():
            continue
        frame_text, separator, prompt = line.partition(":")
        if not separator or not frame_text.strip().isdigit() or not prompt.strip():
            raise ScheduleError(f"invalid prompt keyframe: {line}")
        frame = int(frame_text)
        if frame < 0 or frame >= total_frames:
            raise ScheduleError(f"prompt frame {frame} is outside 0..{total_frames - 1}")
        entries.append((frame, prompt.strip()))
    entries = sorted(dict(entries).items())
    prompts: list[str] = []
    active = entries[0][1]
    cursor = 1
    for frame in range(total_frames):
        while cursor < len(entries) and entries[cursor][0] <= frame:
            active = entries[cursor][1]
            cursor += 1
        prompts.append(active)
    return prompts


@dataclass
class RenderedSchedules:
    prompts: list[str]
    values: dict[str, np.ndarray]

    def frame(self, index: int) -> dict[str, float | str]:
        return {
            "prompt": self.prompts[index],
            **{name: float(values[index]) for name, values in self.values.items()},
        }
