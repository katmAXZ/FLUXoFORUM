"""Versioned public configuration models."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MODEL_ID = "Photoroom/FLUX.2-klein-4b-fp8-diffusers"
BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
CONFIG_VERSION = 1


class Workflow(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class AudioMode(str, Enum):
    ADD = "add"
    MULTIPLY = "multiply"
    REPLACE = "replace"


class AudioMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    feature: Literal["beat", "onset", "rms", "bass", "mids", "highs"]
    mode: AudioMode = AudioMode.ADD
    minimum: float = 0.0
    maximum: float = 1.0
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    smoothing: float = Field(default=0.15, ge=0.0, le=1.0)
    invert: bool = False


class MotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zoom: str = "0:(1.0 + 0.08*f/max(total-1, 1))"
    angle: str = "0:(0.0)"
    translation_x: str = "0:(0.0)"
    translation_y: str = "0:(0.0)"
    perspective_x: str = "0:(0.0)"
    perspective_y: str = "0:(0.0)"
    strength: str = "0:(0.35)"
    noise: str = "0:(0.015)"
    feedback: str = "0:(0.85)"


class FeedbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    color_mode: Literal["LAB", "RGB", "None"] = "LAB"
    reference: Literal["first", "previous", "source"] = "first"
    sharpen: float = Field(default=0.1, ge=0.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.5, le=1.5)
    adaptive_correction: bool = True
    coherent_noise: bool = True
    noise_warp_blend: float = Field(default=0.8, ge=0.0, le=1.0)


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codec: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "medium"
    pixel_format: str = "yuv420p"


class GenerationConfig(BaseModel):
    """Stable JSON contract saved beside every generation."""

    model_config = ConfigDict(extra="forbid")

    config_version: int = CONFIG_VERSION
    workflow: Workflow = Workflow.TEXT
    prompt: str = "A cinematic voyage through a surreal landscape"
    prompt_schedule: str = ""
    negative_prompt: str = ""
    input_image: str | None = None
    input_video: str | None = None
    audio_path: str | None = None
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    frames: int = Field(default=120, ge=1, le=10000)
    fps: float = Field(default=24.0, gt=0.0, le=120.0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    steps: int = Field(default=4, ge=1, le=50)
    guidance: float = Field(default=1.0, ge=0.0, le=20.0)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio_mappings: list[AudioMapping] = Field(default_factory=list)
    optical_flow: Literal["dis", "farneback", "none"] = "dis"
    loader: Literal["auto", "fp8", "bf16"] = "auto"
    model_id: str = MODEL_ID
    base_model_id: str = BASE_MODEL_ID

    @field_validator("width", "height")
    @classmethod
    def dimensions_are_model_compatible(cls, value: int) -> int:
        if value % 16:
            raise ValueError("width and height must be divisible by 16")
        return value

    @field_validator("prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt cannot be blank")
        return value

    @model_validator(mode="after")
    def workflow_has_input(self) -> "GenerationConfig":
        if self.workflow == Workflow.IMAGE and not self.input_image:
            raise ValueError("image workflow requires input_image")
        if self.workflow == Workflow.VIDEO and not self.input_video:
            raise ValueError("video workflow requires input_video")
        if self.audio_mappings and not self.audio_path:
            raise ValueError("audio mappings require audio_path")
        return self

    def resume_signature(self) -> tuple:
        """Fields that cannot change while resuming a partial render."""
        return (
            self.config_version,
            self.workflow.value,
            self.width,
            self.height,
            self.frames,
            self.seed,
            self.steps,
            self.guidance,
            self.model_id,
            self.base_model_id,
            self.model_dump_json(),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GenerationConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
