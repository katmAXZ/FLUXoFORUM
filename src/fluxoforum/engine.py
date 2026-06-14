"""Frame-streaming animation engine."""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .audio import analyze_audio, apply_audio_mappings
from .config import GenerationConfig, Workflow
from .feedback import process_feedback
from .jobs import JobStore, atomic_json
from .model import ImagePipeline, ModelManager
from .noise import CoherentNoise
from .scheduling import RenderedSchedules, parse_prompt_schedule, render_schedule
from .transforms import camera_transform, flow_warp, optical_flow
from .video import VideoReader, encode_video, mux_audio


ProgressCallback = Callable[[int, int, Path | None, str], None]
SCHEDULE_FIELDS = (
    "zoom",
    "angle",
    "translation_x",
    "translation_y",
    "perspective_x",
    "perspective_y",
    "strength",
    "noise",
    "feedback",
)


class RenderCancelled(RuntimeError):
    pass


class AnimationEngine:
    def __init__(
        self,
        store: JobStore,
        model_manager: ModelManager | None = None,
        pipeline: ImagePipeline | None = None,
    ):
        self.store = store
        self.model_manager = model_manager
        self._pipeline = pipeline

    def pipeline(self) -> ImagePipeline:
        if self._pipeline is not None:
            return self._pipeline
        if self.model_manager is None:
            raise RuntimeError("no model manager or pipeline configured")
        return self.model_manager.load()

    def build_schedules(self, config: GenerationConfig) -> tuple[RenderedSchedules, dict]:
        features: dict[str, np.ndarray] = {}
        if config.audio_path:
            features = analyze_audio(config.audio_path, config.fps, config.frames)
        values = {
            name: render_schedule(
                getattr(config.motion, name),
                total_frames=config.frames,
                fps=config.fps,
                audio=features,
            )
            for name in SCHEDULE_FIELDS
        }
        values = apply_audio_mappings(values, features, config.audio_mappings)
        prompts = parse_prompt_schedule(config.prompt_schedule, config.prompt, config.frames)
        return RenderedSchedules(prompts=prompts, values=values), features

    def run(
        self,
        job_id: str,
        resume: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Path:
        try:
            return self._run(job_id, resume=resume, progress=progress)
        except RenderCancelled:
            if self.store.manifest(job_id)["status"] != "cancelled":
                self.store.update(job_id, status="cancelled")
                self.store.append_log(job_id, "Cancelled")
            raise
        except Exception as exc:
            if self.store.manifest(job_id)["status"] != "failed":
                self.store.update(job_id, status="failed", error=str(exc))
                self.store.append_log(job_id, f"Failed: {exc}")
            raise

    def _run(
        self,
        job_id: str,
        resume: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Path:
        paths = self.store.paths(job_id)
        config = self.store.load_config(job_id)
        self.store.reset_cancel(job_id)
        start = 0
        noise_state = None
        previous_generated: Image.Image | None = None
        first_generated: Image.Image | None = None
        if resume:
            self.store.validate_resume(job_id, config)
            if paths.state.exists():
                state = json.loads(paths.state.read_text(encoding="utf-8"))
                if state.get("resume_signature") != list(config.resume_signature()):
                    raise ValueError("resume rejected: saved configuration changed after rendering")
                start = int(state.get("completed_frames", 0))
                noise_state = state.get("noise_state")
            if start:
                previous_generated = Image.open(paths.frames / f"{start - 1:06d}.png").convert("RGB")
                first_generated = Image.open(paths.frames / "000000.png").convert("RGB")

        self._ensure_disk_space(paths.root, config)
        schedules, audio_features = self.build_schedules(config)
        self._validate_inputs(config)
        atomic_json(
            paths.schedules,
            {
                "prompts": schedules.prompts,
                "values": {name: values.tolist() for name, values in schedules.values.items()},
                "audio_features": {
                    name: values.tolist() for name, values in audio_features.items()
                },
            },
        )
        self.store.update(job_id, status="loading_model", completed_frames=start, error=None)
        self._notify(progress, start, config.frames, paths.preview if paths.preview.exists() else None, "Loading model")
        pipe = self.pipeline()
        self.store.append_log(job_id, f"Model ready using {pipe.loader_name}")
        self.store.update(job_id, status="rendering", loader=pipe.loader_name)

        coherent_noise = CoherentNoise(config.seed, config.feedback.noise_warp_blend)
        coherent_noise.restore(noise_state)
        input_image = self._load_input_image(config)
        video_reader = self._open_video(config)
        previous_source: Image.Image | None = None
        if video_reader:
            if start:
                video_reader.seek(max(0, start - 1))
                previous_source = video_reader.read()
            else:
                video_reader.seek(0)

        try:
            for frame_index in range(start, config.frames):
                if self.store.is_cancelled(job_id):
                    raise RenderCancelled("render cancelled by user")
                frame_params = schedules.frame(frame_index)
                source = video_reader.read() if video_reader else None
                if video_reader and source is None:
                    self.store.append_log(job_id, "Input video ended; holding its last frame")
                    source = previous_source
                candidate = self._prepare_candidate(
                    config,
                    frame_index,
                    frame_params,
                    previous_generated,
                    previous_source,
                    source,
                    input_image,
                )
                if candidate is not None and config.feedback.enabled:
                    reference = self._feedback_reference(
                        config, first_generated, previous_generated, source, candidate
                    )
                    noise = (
                        coherent_noise.sample(
                            frame_index,
                            (config.width, config.height),
                            {name: float(frame_params[name]) for name in SCHEDULE_FIELDS},
                        )
                        if config.feedback.coherent_noise
                        else None
                    )
                    candidate = process_feedback(
                        candidate,
                        reference,
                        previous_generated,
                        config.feedback,
                        noise,
                        float(frame_params["noise"]),
                    )
                generated = pipe.generate(
                    prompt=str(frame_params["prompt"]),
                    image=candidate,
                    width=config.width,
                    height=config.height,
                    steps=config.steps,
                    guidance=config.guidance,
                    seed=(config.seed + frame_index) % (2**32),
                    strength=float(frame_params["strength"]),
                )
                frame_path = paths.frames / f"{frame_index:06d}.png"
                generated.save(frame_path, format="PNG", compress_level=3)
                self._save_preview(generated, paths.preview)
                if first_generated is None:
                    first_generated = generated.copy()
                previous_generated = generated
                previous_source = source
                completed = frame_index + 1
                atomic_json(
                    paths.state,
                    {
                        "completed_frames": completed,
                        "noise_state": coherent_noise.serialize(),
                        "resume_signature": list(config.resume_signature()),
                    },
                )
                memory = pipe.memory_stats()
                self._check_vram(memory)
                self.store.update(
                    job_id,
                    status="rendering",
                    completed_frames=completed,
                    preview=str(paths.preview),
                    memory=memory,
                )
                self._notify(progress, completed, config.frames, paths.preview, "Rendering")
                del generated, candidate
                gc.collect()

            self.store.update(job_id, status="encoding")
            self._notify(progress, config.frames, config.frames, paths.preview, "Encoding video")
            output = encode_video(paths.frames, paths.video, config.fps, config.video)
            if config.audio_path:
                output = mux_audio(output, config.audio_path, paths.video_with_audio)
            output = self.store.publish(job_id, output)
            memory = pipe.memory_stats()
            self.store.update(
                job_id,
                status="completed",
                completed_frames=config.frames,
                output=str(output),
                memory=memory,
            )
            self.store.append_log(job_id, f"Completed: {output}")
            self._notify(progress, config.frames, config.frames, paths.preview, "Completed")
            return output
        except RenderCancelled:
            self.store.update(job_id, status="cancelled")
            self.store.append_log(job_id, "Cancelled")
            raise
        except Exception as exc:
            self.store.update(job_id, status="failed", error=str(exc))
            self.store.append_log(job_id, f"Failed: {exc}")
            raise
        finally:
            if video_reader:
                video_reader.close()

    def _prepare_candidate(
        self,
        config: GenerationConfig,
        frame_index: int,
        params: dict,
        previous_generated: Image.Image | None,
        previous_source: Image.Image | None,
        source: Image.Image | None,
        input_image: Image.Image | None,
    ) -> Image.Image | None:
        if frame_index == 0 and config.workflow == Workflow.TEXT:
            return None
        if frame_index == 0 and config.workflow == Workflow.IMAGE:
            return input_image.copy() if input_image else None
        if frame_index == 0 and config.workflow == Workflow.VIDEO:
            return source.copy() if source else None
        if previous_generated is None:
            raise RuntimeError("previous generated frame is unavailable")
        candidate = previous_generated.copy()
        if (
            config.workflow == Workflow.VIDEO
            and config.optical_flow != "none"
            and previous_source is not None
            and source is not None
        ):
            flow = optical_flow(previous_source, source, config.optical_flow)
            candidate = flow_warp(candidate, flow)
        candidate = camera_transform(candidate, params)
        feedback = max(0.0, min(1.0, float(params.get("feedback", 1.0))))
        anchor = source if config.workflow == Workflow.VIDEO else input_image
        if anchor is not None and feedback < 1.0:
            candidate = Image.blend(anchor.resize(candidate.size), candidate, feedback)
        return candidate

    @staticmethod
    def _feedback_reference(
        config: GenerationConfig,
        first: Image.Image | None,
        previous: Image.Image | None,
        source: Image.Image | None,
        fallback: Image.Image,
    ) -> Image.Image:
        if config.feedback.reference == "source" and source is not None:
            return source
        if config.feedback.reference == "previous" and previous is not None:
            return previous
        return first or source or previous or fallback

    @staticmethod
    def _load_input_image(config: GenerationConfig) -> Image.Image | None:
        if not config.input_image:
            return None
        try:
            return Image.open(config.input_image).convert("RGB").resize(
                (config.width, config.height), Image.Resampling.LANCZOS
            )
        except Exception as exc:
            raise ValueError(f"cannot read input image: {exc}") from exc

    @staticmethod
    def _open_video(config: GenerationConfig) -> VideoReader | None:
        if not config.input_video:
            return None
        return VideoReader(config.input_video, config.width, config.height)

    @staticmethod
    def _validate_inputs(config: GenerationConfig) -> None:
        if config.input_image:
            try:
                with Image.open(config.input_image) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"cannot read input image: {exc}") from exc
        if config.input_video:
            reader = VideoReader(config.input_video, config.width, config.height)
            try:
                if reader.read() is None:
                    raise ValueError("input video contains no readable frames")
            finally:
                reader.close()

    @staticmethod
    def _save_preview(image: Image.Image, path: Path) -> None:
        preview = image.copy()
        preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        preview.save(path, format="JPEG", quality=85)

    @staticmethod
    def _ensure_disk_space(path: Path, config: GenerationConfig) -> None:
        free = shutil.disk_usage(path).free
        estimate = config.frames * config.width * config.height * 2
        if free < max(estimate, 512 * 1024**2):
            raise RuntimeError(
                f"insufficient disk space: {free / 1024**3:.1f} GB free, "
                f"approximately {estimate / 1024**3:.1f} GB required"
            )

    @staticmethod
    def _check_vram(memory: dict[str, float]) -> None:
        if memory.get("peak_reserved_gb", 0.0) > 24.0:
            raise RuntimeError(
                f"VRAM safety limit exceeded: {memory['peak_reserved_gb']:.2f} GB reserved"
            )

    @staticmethod
    def _notify(
        callback: ProgressCallback | None,
        completed: int,
        total: int,
        preview: Path | None,
        message: str,
    ) -> None:
        if callback:
            callback(completed, total, preview, message)
