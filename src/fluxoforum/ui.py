"""Gradio application and single-GPU job controller."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .config import AudioMapping, FeedbackConfig, GenerationConfig, MotionConfig, Workflow
from .diagnostics import system_diagnostics
from .engine import AnimationEngine, RenderCancelled
from .jobs import JobStore
from .model import ModelManager


def _file_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "name", None) or str(value)


class AppController:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.model_root = Path(os.getenv("FLUXOFORUM_MODEL_DIR", self.data_root / "models"))
        self.store = JobStore(
            os.getenv("FLUXOFORUM_JOB_DIR", self.data_root / "jobs"),
            os.getenv("FLUXOFORUM_OUTPUT_DIR", self.data_root / "outputs"),
        )
        self.manager: ModelManager | None = None
        self.engine: AnimationEngine | None = None
        self.active_job: str | None = None
        self._render_lock = threading.Lock()

    def configure_engine(self, loader: str) -> AnimationEngine:
        if self.manager is None or self.manager.requested_loader != loader:
            if self.manager:
                self.manager.unload()
            self.manager = ModelManager(self.model_root, requested_loader=loader)
            self.engine = AnimationEngine(self.store, self.manager)
        return self.engine

    def submit(self, config: GenerationConfig):
        if not self._render_lock.acquire(blocking=False):
            raise RuntimeError("another GPU render is already active")
        job_id = self.store.create(config)
        self.active_job = job_id
        engine = self.configure_engine(config.loader)

        updates: list[tuple] = []

        def callback(completed, total, preview, message):
            updates.append(
                (
                    job_id,
                    completed / total if total else 0,
                    str(preview) if preview else None,
                    f"{message}: {completed}/{total}",
                    None,
                )
            )

        try:
            thread_result: dict[str, Any] = {}

            def render():
                try:
                    thread_result["output"] = engine.run(job_id, progress=callback)
                except Exception as exc:
                    thread_result["error"] = exc

            worker = threading.Thread(target=render, daemon=True)
            worker.start()
            while worker.is_alive() or updates:
                if updates:
                    yield updates.pop(0)
                else:
                    worker.join(timeout=0.2)
            if "error" in thread_result:
                error = thread_result["error"]
                if isinstance(error, RenderCancelled):
                    yield job_id, 0, None, "Cancelled", None
                else:
                    raise error
            else:
                output = str(thread_result["output"])
                yield job_id, 1, str(self.store.paths(job_id).preview), "Completed", output
        finally:
            self.active_job = None
            self._render_lock.release()

    def resume(self, job_id: str):
        if not job_id:
            raise ValueError("select or enter a job id")
        if not self._render_lock.acquire(blocking=False):
            raise RuntimeError("another GPU render is already active")
        config = self.store.load_config(job_id)
        engine = self.configure_engine(config.loader)
        self.active_job = job_id
        try:
            output = engine.run(job_id, resume=True)
            return job_id, 1, str(self.store.paths(job_id).preview), "Completed", str(output)
        finally:
            self.active_job = None
            self._render_lock.release()

    def cancel(self) -> str:
        if not self.active_job:
            return "No active render"
        self.store.cancel(self.active_job)
        return f"Cancellation requested for {self.active_job}"

    def history(self) -> list[list[Any]]:
        return [
            [
                job.get("job_id"),
                job.get("status"),
                job.get("completed_frames"),
                job.get("total_frames"),
                job.get("created_at"),
                job.get("output"),
            ]
            for job in self.store.list()
        ]


def create_app(data_root: str | Path | None = None):
    import gradio as gr

    data_root = data_root or os.getenv("FLUXOFORUM_DATA_ROOT", "/workspace/fluxoforum-data")
    controller = AppController(data_root)

    def make_config(
        workflow,
        prompt,
        prompt_schedule,
        input_image,
        input_video,
        audio_path,
        width,
        height,
        frames,
        fps,
        seed,
        zoom,
        angle,
        translation_x,
        translation_y,
        perspective_x,
        perspective_y,
        strength,
        noise,
        feedback_schedule,
        color_mode,
        reference,
        sharpen,
        contrast,
        coherent_noise,
        optical_flow_method,
        steps,
        guidance,
        loader,
        audio_mapping_json,
    ):
        mappings = []
        if audio_mapping_json.strip():
            mappings = [AudioMapping.model_validate(item) for item in json.loads(audio_mapping_json)]
        return GenerationConfig(
            workflow=Workflow(workflow),
            prompt=prompt,
            prompt_schedule=prompt_schedule,
            input_image=_file_path(input_image),
            input_video=_file_path(input_video),
            audio_path=_file_path(audio_path),
            width=int(width),
            height=int(height),
            frames=int(frames),
            fps=float(fps),
            seed=int(seed),
            steps=int(steps),
            guidance=float(guidance),
            loader=loader,
            optical_flow=optical_flow_method,
            motion=MotionConfig(
                zoom=zoom,
                angle=angle,
                translation_x=translation_x,
                translation_y=translation_y,
                perspective_x=perspective_x,
                perspective_y=perspective_y,
                strength=strength,
                noise=noise,
                feedback=feedback_schedule,
            ),
            feedback=FeedbackConfig(
                color_mode=color_mode,
                reference=reference,
                sharpen=float(sharpen),
                contrast=float(contrast),
                coherent_noise=coherent_noise,
            ),
            audio_mappings=mappings,
        )

    def generate(*args):
        config = make_config(*args)
        yield from controller.submit(config)

    def preview_schedules(*args):
        config = make_config(*args)
        engine = controller.engine or AnimationEngine(controller.store, pipeline=None)
        schedules, features = engine.build_schedules(config)
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for name in ("zoom", "angle", "translation_x", "translation_y"):
            axes[0].plot(schedules.values[name], label=name)
        for name in ("strength", "noise", "feedback"):
            axes[1].plot(schedules.values[name], label=name)
        for name, values in features.items():
            axes[2].plot(values, label=name, alpha=0.8)
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("Frame")
        figure.tight_layout()
        return figure

    with gr.Blocks(title="FLUXoFORUM") as app:
        gr.Markdown("# FLUXoFORUM\nDeforum-style animation with FLUX.2 Klein FP8")
        job_id = gr.Textbox(label="Job ID", interactive=False)
        with gr.Tab("Generate"):
            workflow = gr.Dropdown(
                [item.value for item in Workflow], value=Workflow.TEXT.value, label="Workflow"
            )
            prompt = gr.Textbox(
                value="A cinematic voyage through a surreal landscape",
                lines=3,
                label="Prompt",
            )
            prompt_schedule = gr.Textbox(
                label="Prompt keyframes",
                placeholder="0: surreal forest\n60: crystalline city",
                lines=3,
            )
            with gr.Row():
                input_image = gr.Image(type="filepath", label="Init image")
                input_video = gr.Video(label="Input video")
                audio_path = gr.Audio(type="filepath", label="Audio")
            with gr.Row():
                width = gr.Number(1024, precision=0, label="Width")
                height = gr.Number(1024, precision=0, label="Height")
                frames = gr.Number(120, precision=0, label="Frames")
                fps = gr.Number(24, label="FPS")
                seed = gr.Number(42, precision=0, label="Seed")
        with gr.Tab("Motion"):
            gr.Markdown("Use `frame:(expression)` schedules. Variables: `f`, `t`, audio bands.")
            zoom = gr.Textbox("0:(1.0 + 0.08*f/max(total-1, 1))", label="Zoom")
            angle = gr.Textbox("0:(0.0)", label="Angle")
            translation_x = gr.Textbox("0:(0.0)", label="Translation X")
            translation_y = gr.Textbox("0:(0.0)", label="Translation Y")
            perspective_x = gr.Textbox("0:(0.0)", label="Perspective X")
            perspective_y = gr.Textbox("0:(0.0)", label="Perspective Y")
            strength = gr.Textbox("0:(0.35)", label="Denoise strength")
            noise = gr.Textbox("0:(0.015)", label="Noise")
            feedback_schedule = gr.Textbox("0:(0.85)", label="Feedback")
            schedule_preview_button = gr.Button("Preview curves")
            schedule_plot = gr.Plot(label="Rendered schedules")
        with gr.Tab("Audio"):
            audio_mapping_json = gr.Code(
                value='[\n  {"target":"zoom","feature":"bass","mode":"add","minimum":0.0,"maximum":0.01,"strength":1.0}\n]',
                language="json",
                label="Audio mappings",
            )
            gr.Markdown(
                "Features: `beat`, `onset`, `rms`, `bass`, `mids`, `highs`. "
                "Modes: `add`, `multiply`, `replace`."
            )
        with gr.Tab("Advanced"):
            with gr.Row():
                color_mode = gr.Dropdown(["LAB", "RGB", "None"], value="LAB", label="Color")
                reference = gr.Dropdown(
                    ["first", "previous", "source"], value="first", label="Reference"
                )
                optical_flow_method = gr.Dropdown(
                    ["dis", "farneback", "none"], value="dis", label="Optical flow"
                )
                loader = gr.Dropdown(["auto", "fp8", "bf16"], value="auto", label="Loader")
            with gr.Row():
                sharpen = gr.Slider(0, 0.5, 0.1, label="Sharpen")
                contrast = gr.Slider(0.5, 1.5, 1.0, label="Contrast")
                coherent_noise = gr.Checkbox(True, label="Coherent noise")
                steps = gr.Number(4, precision=0, label="Steps")
                guidance = gr.Number(1.0, label="Guidance")
        with gr.Tab("Jobs / System"):
            diagnostics = gr.JSON(system_diagnostics(data_root), label="System diagnostics")
            refresh_diagnostics = gr.Button("Refresh diagnostics")
            history = gr.Dataframe(
                headers=["job_id", "status", "done", "total", "created", "output"],
                value=controller.history(),
                interactive=False,
            )
            with gr.Row():
                history_refresh = gr.Button("Refresh history")
                resume_id = gr.Textbox(label="Job ID to resume")
                resume_button = gr.Button("Resume")
        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary")
            cancel_button = gr.Button("Cancel")
        progress = gr.Slider(0, 1, 0, interactive=False, label="Progress")
        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Image(label="Live preview", interactive=False)
        output = gr.Video(label="Output", interactive=False)

        config_inputs = [
            workflow,
            prompt,
            prompt_schedule,
            input_image,
            input_video,
            audio_path,
            width,
            height,
            frames,
            fps,
            seed,
            zoom,
            angle,
            translation_x,
            translation_y,
            perspective_x,
            perspective_y,
            strength,
            noise,
            feedback_schedule,
            color_mode,
            reference,
            sharpen,
            contrast,
            coherent_noise,
            optical_flow_method,
            steps,
            guidance,
            loader,
            audio_mapping_json,
        ]
        generate_button.click(
            generate,
            inputs=config_inputs,
            outputs=[job_id, progress, preview, status, output],
        )
        schedule_preview_button.click(
            preview_schedules, inputs=config_inputs, outputs=schedule_plot
        )
        cancel_button.click(controller.cancel, outputs=status)
        history_refresh.click(controller.history, outputs=history)
        refresh_diagnostics.click(
            lambda loader_value: system_diagnostics(data_root, loader_value),
            inputs=loader,
            outputs=diagnostics,
        )
        resume_button.click(
            controller.resume,
            inputs=resume_id,
            outputs=[job_id, progress, preview, status, output],
        )
    return app
