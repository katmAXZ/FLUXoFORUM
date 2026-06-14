from __future__ import annotations

import os
from pathlib import Path

import pytest

from fluxoforum.config import FeedbackConfig, GenerationConfig
from fluxoforum.engine import AnimationEngine
from fluxoforum.jobs import JobStore
from fluxoforum.model import ModelManager


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.getenv("FLUXOFORUM_GPU_TEST") != "1",
    reason="set FLUXOFORUM_GPU_TEST=1 on a configured RunPod",
)
def test_real_model_16_frame_vram(tmp_path, monkeypatch):
    manager = ModelManager(
        os.getenv("FLUXOFORUM_MODEL_DIR", "/workspace/fluxoforum-data/models")
    )
    store = JobStore(tmp_path / "jobs", tmp_path / "outputs")
    config = GenerationConfig(
        width=1024,
        height=1024,
        frames=16,
        feedback=FeedbackConfig(enabled=False),
    )
    job_id = store.create(config)

    def fake_encode(frames_dir, output, fps, video_config):
        output = Path(output)
        output.write_bytes(b"gpu smoke")
        return output

    monkeypatch.setattr("fluxoforum.engine.encode_video", fake_encode)
    AnimationEngine(store, manager).run(job_id)
    memory = store.manifest(job_id)["memory"]
    assert memory["peak_reserved_gb"] <= 24.0
