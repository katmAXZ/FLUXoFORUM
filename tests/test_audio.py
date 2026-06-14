import numpy as np

from fluxoforum.audio import apply_audio_mappings
from fluxoforum.config import AudioMapping


def test_audio_add_mapping():
    schedules = {"zoom": np.ones(3)}
    features = {"bass": np.array([0.0, 0.5, 1.0])}
    mapping = AudioMapping(
        target="zoom",
        feature="bass",
        mode="add",
        minimum=0,
        maximum=0.1,
        smoothing=0,
    )
    result = apply_audio_mappings(schedules, features, [mapping])
    assert np.allclose(result["zoom"], [1.0, 1.05, 1.1])

