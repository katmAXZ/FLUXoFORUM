"""Startup and health diagnostics."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .config import BASE_MODEL_ID, MODEL_ID
from .model import select_loader


def system_diagnostics(
    data_root: str | Path,
    requested_loader: str = "auto",
    check_model_access: bool = False,
) -> dict[str, Any]:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    result: dict[str, Any] = {
        "data_root": str(root),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "hf_token_set": bool(os.getenv("HF_TOKEN")),
        "cuda_available": False,
        "selected_loader": "unknown",
        "estimated_peak_vram_gb": "18-22 (FP8) / compatibility-dependent (BF16)",
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            decision = select_loader(requested_loader, True, capability)
            properties = torch.cuda.get_device_properties(0)
            result.update(
                {
                    "gpu": properties.name,
                    "gpu_vram_gb": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                    "selected_loader": decision.selected,
                    "loader_reason": decision.reason,
                }
            )
        else:
            result["loader_reason"] = "CUDA is required for generation"
    except ImportError:
        result["loader_reason"] = "PyTorch is not installed"
    if check_model_access:
        result["model_access"] = validate_model_access()
    return result


def health(data_root: str | Path) -> tuple[bool, dict[str, Any]]:
    report = system_diagnostics(data_root)
    ok = report["disk_free_gb"] >= 1 and Path(data_root).exists()
    return ok, report


def validate_model_access() -> dict[str, str]:
    try:
        from huggingface_hub import model_info

        token = os.getenv("HF_TOKEN")
        model_info(MODEL_ID, token=token)
        model_info(BASE_MODEL_ID, token=token)
        return {"status": "ok", "message": "Both Hugging Face repositories are accessible"}
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Model access failed. Accept the BFL license and set HF_TOKEN. "
                f"Details: {exc}"
            ),
        }
