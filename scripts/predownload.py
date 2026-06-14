"""Download model assets to the persistent RunPod volume without loading CUDA."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from fluxoforum.config import BASE_MODEL_ID, MODEL_ID


def main() -> None:
    root = Path(os.getenv("FLUXOFORUM_MODEL_DIR", "/workspace/fluxoforum-data/models"))
    root.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required after accepting the BFL model terms")
    print("Downloading Photoroom FP8 and BF16 fallback transformer...")
    snapshot_download(
        MODEL_ID,
        cache_dir=root,
        token=token,
        allow_patterns=[
            "transformer_fp8_static/*",
            "transformer_bf16/*",
            "*.json",
            "README.md",
        ],
    )
    print("Downloading BFL text encoder, VAE, tokenizer, and scheduler...")
    snapshot_download(BASE_MODEL_ID, cache_dir=root, token=token)
    print(f"Models cached under {root}")


if __name__ == "__main__":
    main()

