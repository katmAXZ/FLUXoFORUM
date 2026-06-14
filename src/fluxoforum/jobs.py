"""Persistent, atomic render job storage."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import GenerationConfig


TERMINAL_STATES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def schedules(self) -> Path:
        return self.root / "schedules.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def logs(self) -> Path:
        return self.root / "render.log"

    @property
    def preview(self) -> Path:
        return self.root / "preview.jpg"

    @property
    def video(self) -> Path:
        return self.root / "output.mp4"

    @property
    def video_with_audio(self) -> Path:
        return self.root / "output_audio.mp4"


class JobStore:
    def __init__(self, root: str | Path, outputs_root: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs_root = Path(outputs_root) if outputs_root else self.root.parent / "outputs"
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def create(self, config: GenerationConfig) -> str:
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        paths = self.paths(job_id)
        paths.frames.mkdir(parents=True)
        config.to_json_file(paths.config)
        atomic_json(
            paths.manifest,
            {
                "job_id": job_id,
                "status": "queued",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "completed_frames": 0,
                "total_frames": config.frames,
                "error": None,
                "output": None,
            },
        )
        self._cancel_events[job_id] = threading.Event()
        return job_id

    def paths(self, job_id: str) -> JobPaths:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", job_id):
            raise ValueError("invalid job id")
        path = (self.root / job_id).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("job path escapes store")
        return JobPaths(path)

    def load_config(self, job_id: str) -> GenerationConfig:
        return GenerationConfig.from_json_file(self.paths(job_id).config)

    def manifest(self, job_id: str) -> dict[str, Any]:
        return json.loads(self.paths(job_id).manifest.read_text(encoding="utf-8"))

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            manifest = self.manifest(job_id)
            manifest.update(changes)
            manifest["updated_at"] = utc_now()
            atomic_json(self.paths(job_id).manifest, manifest)
            return manifest

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            with self.paths(job_id).logs.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utc_now()}] {message}\n")

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        jobs = []
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                jobs.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda job: job.get("created_at", ""), reverse=True)[:limit]

    def cancel(self, job_id: str) -> None:
        event = self._cancel_events.setdefault(job_id, threading.Event())
        event.set()
        manifest = self.manifest(job_id)
        if manifest["status"] not in TERMINAL_STATES:
            self.update(job_id, status="cancelling")

    def is_cancelled(self, job_id: str) -> bool:
        return self._cancel_events.setdefault(job_id, threading.Event()).is_set()

    def reset_cancel(self, job_id: str) -> None:
        self._cancel_events[job_id] = threading.Event()

    def validate_resume(self, job_id: str, config: GenerationConfig) -> None:
        stored = self.load_config(job_id)
        if stored.resume_signature() != config.resume_signature():
            raise ValueError("resume rejected: model, dimensions, frame count, or schedules changed")

    def publish(self, job_id: str, source: str | Path) -> Path:
        source = Path(source)
        destination = self.outputs_root / f"{job_id}{source.suffix}"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return destination
