# FLUXoFORUM

FLUXoFORUM is a standalone, persistent RunPod Gradio application for Deforum-style
animation with FLUX.2 Klein 4B. It streams one frame at a time, persists every
render step, and is designed for NVIDIA GPUs with at most 24 GB VRAM.

## Features

- Text-to-animation, init-image animation, and input-video stylization
- Deforum `frame:(expression)` schedules with safe arithmetic and oscillators
- Prompt keyframes, camera transforms, optical-flow transfer, and coherent noise
- LAB/RGB color stabilization and adaptive blur/burn correction
- Audio analysis for beats, onset, RMS, bass, mids, and highs
- Persistent jobs with cancellation, history, atomic checkpoints, and resume
- Automatic Photoroom static FP8 loading with a BF16/offload compatibility fallback
- Gradio UI on port `7860`, optional basic authentication, and RunPod assets

## Model Layout

The preferred transformer is `Photoroom/FLUX.2-klein-4b-fp8-diffusers`. That
repository contains transformer weights only. FLUXoFORUM loads the text encoder,
tokenizer, VAE, and scheduler from `black-forest-labs/FLUX.2-klein-4B`.

The application first attempts the approximately 3.9 GB TorchAO static-FP8
transformer. If deserialization or GPU capability is incompatible, it loads
Photoroom's BF16 transformer and enables Diffusers model CPU offloading.

## RunPod

1. Create a persistent Pod with a 24 GB NVIDIA GPU and attach a network volume
   at `/workspace`.
2. Expose HTTP port `7860`.
3. Accept the BFL Klein 4B terms on Hugging Face and set `HF_TOKEN`.
4. Run:

```bash
cd /workspace/FLUXoFORUM
bash scripts/setup_runpod.sh
source /workspace/fluxoforum-venv/bin/activate
python scripts/predownload.py
bash scripts/launch.sh
```

If an earlier installation failed while resolving `safetensors`, update the
project files and run `bash scripts/setup_runpod.sh` again. FLUXoFORUM pins
Diffusers `0.37.1`, the latest tested release that includes
`Flux2KleinPipeline` without requiring an unreleased safetensors package.

Persistent paths default to:

```text
/workspace/fluxoforum-data/models
/workspace/fluxoforum-data/jobs
/workspace/fluxoforum-data/outputs
```

Set `FLUXOFORUM_USERNAME` and `FLUXOFORUM_PASSWORD` to protect Gradio.

## Docker

```bash
docker build -t fluxoforum:1.0 .
docker run --gpus all -p 7860:7860 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /workspace/fluxoforum-data:/workspace/fluxoforum-data \
  fluxoforum:1.0
```

## Schedules

Numeric schedules use comma-separated `frame:(expression)` entries:

```text
0:(1.0), 60:(1.08), 119:(1.0)
```

Expressions may use `f`, `t`, `fps`, `total`, audio feature names, arithmetic,
and `sin`, `cos`, `tan`, `sqrt`, `abs`, `min`, `max`, `clamp`, or `lerp`.

Prompt schedules use one entry per line:

```text
0: a misty ancient forest
60: a crystalline city at sunrise
```

Audio mappings are JSON:

```json
[
  {
    "target": "zoom",
    "feature": "bass",
    "mode": "add",
    "minimum": 0.0,
    "maximum": 0.01,
    "strength": 1.0
  }
]
```

## Recovery

Every job stores validated config, rendered schedules, individual PNG frames,
latest noise state, logs, preview, and output video. Resume is allowed only when
the saved model, dimensions, frame count, and schedules are unchanged.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

The default 1024x1024, 120-frame profile has a hard 24 GB reserved-VRAM guard.
The operational target is below 22 GB.
