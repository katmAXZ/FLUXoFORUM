from pathlib import Path

import pytest
from PIL import Image

from fluxoforum.config import FeedbackConfig, GenerationConfig, Workflow
from fluxoforum.engine import AnimationEngine
from fluxoforum.jobs import JobStore


def _fake_encode(frames_dir, output, fps, config):
    output = Path(output)
    output.write_bytes(b"fake mp4")
    return output


def test_text_animation(fake_pipeline, tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    config = GenerationConfig(
        width=256,
        height=256,
        frames=3,
        feedback=FeedbackConfig(enabled=False),
    )
    job_id = store.create(config)
    engine = AnimationEngine(store, pipeline=fake_pipeline)
    monkeypatch.setattr("fluxoforum.engine.encode_video", _fake_encode)
    output = engine.run(job_id)
    assert output.exists()
    assert len(fake_pipeline.calls) == 3
    assert fake_pipeline.calls[0]["has_image"] is False
    assert fake_pipeline.calls[1]["has_image"] is True
    assert store.manifest(job_id)["status"] == "completed"


def test_image_animation(fake_pipeline, tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (256, 256), "navy").save(image_path)
    store = JobStore(tmp_path / "jobs")
    config = GenerationConfig(
        workflow=Workflow.IMAGE,
        input_image=str(image_path),
        width=256,
        height=256,
        frames=2,
        feedback=FeedbackConfig(enabled=False),
    )
    job_id = store.create(config)
    monkeypatch.setattr("fluxoforum.engine.encode_video", _fake_encode)
    AnimationEngine(store, pipeline=fake_pipeline).run(job_id)
    assert all(call["has_image"] for call in fake_pipeline.calls)


def test_video_animation(fake_pipeline, tmp_path, monkeypatch):
    class FakeVideoReader:
        def __init__(self, *args):
            self.index = 0

        def seek(self, frame):
            self.index = frame

        def read(self):
            if self.index >= 2:
                return None
            image = Image.new("RGB", (256, 256), (self.index * 30, 20, 40))
            self.index += 1
            return image

        def close(self):
            return None

    store = JobStore(tmp_path / "jobs")
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    config = GenerationConfig(
        workflow=Workflow.VIDEO,
        input_video=str(video_path),
        width=256,
        height=256,
        frames=2,
        optical_flow="none",
        feedback=FeedbackConfig(enabled=False),
    )
    job_id = store.create(config)
    monkeypatch.setattr("fluxoforum.engine.VideoReader", FakeVideoReader)
    monkeypatch.setattr("fluxoforum.engine.encode_video", _fake_encode)
    AnimationEngine(store, pipeline=fake_pipeline).run(job_id)
    assert all(call["has_image"] for call in fake_pipeline.calls)


def test_resume(fake_pipeline, tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    config = GenerationConfig(
        width=256,
        height=256,
        frames=3,
        feedback=FeedbackConfig(enabled=False),
    )
    job_id = store.create(config)
    engine = AnimationEngine(store, pipeline=fake_pipeline)
    monkeypatch.setattr("fluxoforum.engine.encode_video", _fake_encode)
    original = fake_pipeline.generate

    def stop_after_first(*args, **kwargs):
        if len(fake_pipeline.calls) == 1:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    fake_pipeline.generate = stop_after_first
    try:
        engine.run(job_id)
    except RuntimeError:
        pass
    fake_pipeline.generate = original
    output = engine.run(job_id, resume=True)
    assert output.exists()
    assert store.manifest(job_id)["status"] == "completed"


def test_cancel_during_render(fake_pipeline, tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    job_id = store.create(
        GenerationConfig(
            width=256,
            height=256,
            frames=3,
            feedback=FeedbackConfig(enabled=False),
        )
    )
    monkeypatch.setattr("fluxoforum.engine.encode_video", _fake_encode)

    def progress(completed, total, preview, message):
        if completed == 1:
            store.cancel(job_id)

    with pytest.raises(Exception, match="cancelled"):
        AnimationEngine(store, pipeline=fake_pipeline).run(job_id, progress=progress)
    assert store.manifest(job_id)["status"] == "cancelled"


def test_resume_rejects_modified_config(fake_pipeline, tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    config = GenerationConfig(
        width=256,
        height=256,
        frames=3,
        feedback=FeedbackConfig(enabled=False),
    )
    job_id = store.create(config)
    engine = AnimationEngine(store, pipeline=fake_pipeline)
    monkeypatch.setattr("fluxoforum.engine.encode_video", _fake_encode)
    original = fake_pipeline.generate

    def interrupt(*args, **kwargs):
        if len(fake_pipeline.calls) == 1:
            raise RuntimeError("interrupt")
        return original(*args, **kwargs)

    fake_pipeline.generate = interrupt
    with pytest.raises(RuntimeError, match="interrupt"):
        engine.run(job_id)
    fake_pipeline.generate = original
    changed = store.load_config(job_id).model_copy(update={"seed": 99})
    changed.to_json_file(store.paths(job_id).config)
    with pytest.raises(ValueError, match="configuration changed"):
        engine.run(job_id, resume=True)


def test_corrupt_image_fails_before_model(fake_pipeline, tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")
    store = JobStore(tmp_path / "jobs")
    job_id = store.create(
        GenerationConfig(
            workflow=Workflow.IMAGE,
            input_image=str(image_path),
            width=256,
            height=256,
            frames=1,
        )
    )
    with pytest.raises(ValueError, match="cannot read input image"):
        AnimationEngine(store, pipeline=fake_pipeline).run(job_id)
    assert fake_pipeline.calls == []
