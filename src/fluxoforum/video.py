"""Video input, frame encoding, and audio muxing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

from .config import VideoConfig


class VideoReader:
    def __init__(self, path: str | Path, width: int, height: int):
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"cannot open input video: {path}")
        self.width = width
        self.height = height

    def read(self) -> Image.Image | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).resize((self.width, self.height), Image.Resampling.LANCZOS)

    def seek(self, frame: int) -> None:
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame)

    def close(self) -> None:
        self.capture.release()


def ffmpeg_executable() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg is required but was not found") from exc


def encode_video(
    frames_dir: str | Path,
    output: str | Path,
    fps: float,
    config: VideoConfig,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(Path(frames_dir) / "%06d.png"),
        "-c:v",
        config.codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        config.pixel_format,
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg encoding failed: {completed.stderr[-2000:]}")
    return output


def mux_audio(video: str | Path, audio: str | Path, output: str | Path) -> Path:
    command = [
        ffmpeg_executable(),
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg audio mux failed: {completed.stderr[-2000:]}")
    return Path(output)
