from fluxoforum.config import GenerationConfig
from fluxoforum.jobs import JobStore


def test_job_round_trip(tmp_path):
    store = JobStore(tmp_path)
    config = GenerationConfig(width=256, height=256, frames=2)
    job_id = store.create(config)
    assert store.load_config(job_id) == config
    assert store.manifest(job_id)["status"] == "queued"
    store.update(job_id, status="rendering", completed_frames=1)
    assert store.manifest(job_id)["completed_frames"] == 1


def test_cancel(tmp_path):
    store = JobStore(tmp_path)
    job_id = store.create(GenerationConfig(width=256, height=256, frames=1))
    store.cancel(job_id)
    assert store.is_cancelled(job_id)

