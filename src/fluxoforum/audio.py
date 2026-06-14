"""Audio analysis and schedule modulation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .config import AudioMapping, AudioMode


FEATURE_NAMES = ("beat", "onset", "rms", "bass", "mids", "highs")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float64))
    low, high = np.percentile(values, [2, 98])
    if high - low < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def resample(values: np.ndarray, frames: int) -> np.ndarray:
    if len(values) == frames:
        return values
    if not len(values):
        return np.zeros(frames, dtype=np.float64)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, frames)
    return np.interp(target, source, values)


def smooth(values: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0 or len(values) < 2:
        return values
    radius = max(1, int(round(amount * 12)))
    kernel = np.ones(radius * 2 + 1, dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def analyze_audio(path: str | Path, fps: float, frames: int) -> dict[str, np.ndarray]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("audio analysis requires librosa") from exc

    waveform, sample_rate = librosa.load(str(path), sr=22050, mono=True)
    if waveform.size == 0:
        raise ValueError("audio file is empty")
    hop_length = max(64, round(sample_rate / fps))
    stft = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    def band(low: float, high: float) -> np.ndarray:
        mask = (frequencies >= low) & (frequencies < high)
        return stft[mask].mean(axis=0) if mask.any() else np.zeros(stft.shape[1])

    rms = librosa.feature.rms(S=stft, frame_length=2048, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(
        y=waveform, sr=sample_rate, hop_length=hop_length, aggregate=np.median
    )
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sample_rate, hop_length=hop_length
    )
    beat = np.zeros_like(onset)
    beat[np.asarray(beat_frames, dtype=int).clip(0, max(0, len(beat) - 1))] = 1.0

    features = {
        "beat": beat,
        "onset": onset,
        "rms": rms,
        "bass": band(20, 250),
        "mids": band(250, 4000),
        "highs": band(4000, 10000),
    }
    return {name: resample(normalize(values), frames) for name, values in features.items()}


def apply_audio_mappings(
    schedules: dict[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    mappings: list[AudioMapping],
) -> dict[str, np.ndarray]:
    result = {name: values.copy() for name, values in schedules.items()}
    for mapping in mappings:
        if mapping.target not in result or mapping.feature not in features:
            continue
        audio = smooth(np.asarray(features[mapping.feature]), mapping.smoothing)
        audio = np.where(audio >= mapping.threshold, audio, 0.0)
        if mapping.invert:
            audio = 1.0 - audio
        scaled = mapping.minimum + audio * (mapping.maximum - mapping.minimum)
        base = result[mapping.target]
        if mapping.mode == AudioMode.REPLACE:
            result[mapping.target] = base + (scaled - base) * mapping.strength
        elif mapping.mode == AudioMode.MULTIPLY:
            result[mapping.target] = base * (1.0 + (scaled - 1.0) * mapping.strength)
        else:
            result[mapping.target] = base + scaled * mapping.strength
    return result

