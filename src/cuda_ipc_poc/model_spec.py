"""Model specification, weight loading, and registry.

ModelSpec bundles everything needed to load, shard, and run a model:
  - model_cls / model_factory: how to create the model shell
  - sample_input_fn: test input generator
  - tp_handler: optional TP sharding rules
  - weight_loader: optional external weight loading (e.g., safetensors)

ModelRegistry is the central lookup. Modules register their specs at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .tp_handler import BaseTensorParallelHandler


@runtime_checkable
class WeightLoader(Protocol):
    """Protocol for loading model weights from external files."""

    def load(self, model_path: str, device: str) -> dict[str, torch.Tensor]: ...


class SafetensorsLoader:
    """Load state_dict from HuggingFace safetensors files."""

    def load(self, model_path: str, device: str) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        return load_file(model_path, device=device)


@dataclass
class ModelSpec:
    """Complete specification for a model in the CUDA IPC pipeline."""

    model_cls: type[nn.Module]
    model_factory: Callable[..., nn.Module]
    sample_input_fn: Callable[..., torch.Tensor]
    tp_handler: BaseTensorParallelHandler | None = None
    weight_loader: WeightLoader | None = None


class ModelRegistry:
    """Global registry mapping model names to ModelSpec instances."""

    _specs: dict[str, ModelSpec] = {}

    @classmethod
    def register(cls, name: str, spec: ModelSpec) -> None:
        cls._specs[name] = spec

    @classmethod
    def get(cls, name: str) -> ModelSpec:
        if name not in cls._specs:
            available = ", ".join(sorted(cls._specs)) or "(none)"
            raise KeyError(
                f"Unknown model: {name!r}. Available: {available}"
            )
        return cls._specs[name]

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._specs.keys())

    @classmethod
    def _clear(cls) -> None:
        """For testing only — reset the registry."""
        cls._specs.clear()
