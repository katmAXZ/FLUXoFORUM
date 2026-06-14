"""Memory-aware FLUX.2 Klein model adapter."""

from __future__ import annotations

import gc
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .config import BASE_MODEL_ID, MODEL_ID

logger = logging.getLogger(__name__)


class ImagePipeline(Protocol):
    loader_name: str

    def generate(
        self,
        prompt: str,
        image: Image.Image | None,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        strength: float,
    ) -> Image.Image: ...

    def memory_stats(self) -> dict[str, float]: ...


@dataclass
class LoaderDecision:
    requested: str
    selected: str
    reason: str
    compute_capability: tuple[int, int] | None


def select_loader(requested: str, cuda_available: bool, capability: tuple[int, int] | None) -> LoaderDecision:
    if requested == "bf16":
        return LoaderDecision(requested, "bf16", "BF16 explicitly requested", capability)
    if not cuda_available:
        return LoaderDecision(requested, "bf16", "CUDA unavailable; model load will fail clearly", capability)
    fp8_supported = capability is not None and capability >= (8, 9)
    if requested == "fp8" and not fp8_supported:
        return LoaderDecision(requested, "bf16", "GPU lacks practical FP8 support", capability)
    if fp8_supported:
        return LoaderDecision(requested, "fp8", "modern CUDA GPU detected", capability)
    return LoaderDecision(requested, "bf16", "using compatibility loader", capability)


def load_torchao_fp8_static_model(
    checkpoint: str | Path,
    factory,
    device: str = "cuda",
    strict: bool = True,
):
    """Restore Photoroom's static activation/weight FP8 transformer."""
    import torch
    import torch.nn as nn
    from torchao.quantization import (
        Float8StaticActivationFloat8WeightConfig,
        PerTensor,
        quantize_,
    )

    try:
        from torchao.quantization import FqnToConfig
    except ImportError:
        from torchao.quantization import ModuleFqnToConfig as FqnToConfig

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    required = {"state_dict", "act_scales", "fp8_dtype"}
    if not required.issubset(payload):
        raise ValueError(f"FP8 checkpoint missing keys: {sorted(required - set(payload))}")

    dtype_text = str(payload["fp8_dtype"])
    if "float8_e4m3fn" in dtype_text:
        fp8_dtype = torch.float8_e4m3fn
    elif "float8_e5m2" in dtype_text:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"unsupported FP8 dtype: {dtype_text}")

    raw_scales = {
        name: (
            value.detach().to(torch.float32).reshape(-1)[0]
            if torch.is_tensor(value)
            else torch.tensor(float(value), dtype=torch.float32)
        )
        for name, value in payload["act_scales"].items()
    }
    model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError("FP8 model factory must return torch.nn.Module")
    model.eval().to(device)

    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    linear_set = set(linear_names)
    candidates = [
        raw_scales,
        {name[6:]: value for name, value in raw_scales.items() if name.startswith("model.")},
        {f"model.{name}": value for name, value in raw_scales.items()},
    ]
    scales = max(candidates, key=lambda item: sum(name in linear_set for name in item))
    if not any(name in linear_set for name in scales):
        raise RuntimeError("FP8 activation-scale names do not match transformer Linear layers")

    config_by_name = {
        name: Float8StaticActivationFloat8WeightConfig(
            scale=scales[name],
            activation_dtype=fp8_dtype,
            weight_dtype=fp8_dtype,
            granularity=PerTensor(),
        )
        for name in linear_names
        if name in scales
    }
    try:
        quantization_config = FqnToConfig(fqn_to_config=config_by_name)
    except TypeError:
        quantization_config = FqnToConfig(config_by_name)
    quantize_(model, quantization_config, filter_fn=None, device=device)

    try:
        missing, unexpected = model.load_state_dict(
            payload["state_dict"], strict=strict, assign=True
        )
    except TypeError:
        modules = dict(model.named_modules())
        for name, tensor in payload["state_dict"].items():
            module_name, attribute = name.rsplit(".", 1)
            module = modules[module_name]
            current = getattr(module, attribute)
            if isinstance(current, nn.Parameter):
                setattr(module, attribute, nn.Parameter(tensor, requires_grad=False))
            else:
                setattr(module, attribute, tensor)
        missing, unexpected = [], []
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"FP8 state mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return model


class KleinPipeline:
    def __init__(self, pipe: Any, loader_name: str, torch_module: Any):
        self.pipe = pipe
        self.loader_name = loader_name
        self.torch = torch_module
        self._call_parameters = set(inspect.signature(pipe.__call__).parameters)

    def generate(
        self,
        prompt: str,
        image: Image.Image | None,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        strength: float,
    ) -> Image.Image:
        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": generator,
        }
        if "image" in self._call_parameters:
            kwargs["image"] = [image] if image is not None else [None]
        if image is not None and "strength" in self._call_parameters:
            kwargs["strength"] = strength
        with self.torch.inference_mode():
            result = self.pipe(**kwargs).images[0].convert("RGB")
        if image is not None and "strength" not in self._call_parameters and strength < 0.999:
            result = Image.blend(image.resize(result.size), result, max(0.0, min(1.0, strength)))
        return result

    def memory_stats(self) -> dict[str, float]:
        if not self.torch.cuda.is_available():
            return {}
        gib = 1024**3
        return {
            "allocated_gb": self.torch.cuda.memory_allocated() / gib,
            "reserved_gb": self.torch.cuda.memory_reserved() / gib,
            "peak_allocated_gb": self.torch.cuda.max_memory_allocated() / gib,
            "peak_reserved_gb": self.torch.cuda.max_memory_reserved() / gib,
        }


class ModelManager:
    def __init__(
        self,
        model_dir: str | Path,
        requested_loader: str = "auto",
        model_id: str = MODEL_ID,
        base_model_id: str = BASE_MODEL_ID,
    ):
        self.model_dir = Path(model_dir)
        self.requested_loader = requested_loader
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.pipeline: KleinPipeline | None = None
        self.decision: LoaderDecision | None = None

    def load(self) -> KleinPipeline:
        if self.pipeline is not None:
            return self.pipeline
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        if not torch.cuda.is_available():
            raise RuntimeError("FLUXoFORUM requires an NVIDIA CUDA GPU")
        capability = torch.cuda.get_device_capability()
        self.decision = select_loader(self.requested_loader, True, capability)
        os.environ.setdefault("HF_HOME", str(self.model_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.model_dir / "hub"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        transformer = None
        if self.decision.selected == "fp8":
            try:
                from huggingface_hub import hf_hub_download

                checkpoint = hf_hub_download(
                    self.model_id,
                    filename="transformer_fp8_static/model_fp8_static.pt",
                    cache_dir=str(self.model_dir),
                )
                transformer = load_torchao_fp8_static_model(
                    checkpoint,
                    factory=lambda: Flux2Transformer2DModel.from_pretrained(
                        self.model_id,
                        subfolder="transformer_bf16",
                        torch_dtype=torch.bfloat16,
                        cache_dir=str(self.model_dir),
                    ),
                )
            except Exception as exc:
                if self.requested_loader == "fp8":
                    logger.warning("FP8 load failed; using BF16 fallback: %s", exc)
                self.decision = LoaderDecision(
                    self.requested_loader, "bf16", f"FP8 initialization failed: {exc}", capability
                )
                transformer = None

        if transformer is None:
            transformer = Flux2Transformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer_bf16",
                torch_dtype=torch.bfloat16,
                cache_dir=str(self.model_dir),
            )

        pipe = Flux2KleinPipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=str(self.model_dir),
        )
        if hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self.pipeline = KleinPipeline(pipe, self.decision.selected, torch)
        return self.pipeline

    def unload(self) -> None:
        self.pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
