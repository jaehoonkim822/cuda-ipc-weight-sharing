"""Test model definitions for CUDA IPC PoC."""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Minimal MLP: 784 -> 256 -> 10.

    4 parameter tensors total (fc1.weight, fc1.bias, fc2.weight, fc2.bias).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _mlp_sample_input(device: str, batch_size: int = 1) -> torch.Tensor:
    return torch.randn(batch_size, 784, device=device)


def _resnet_sample_input(device: str, batch_size: int = 1) -> torch.Tensor:
    return torch.randn(batch_size, 3, 224, 224, device=device)


def get_model(name: str = "mlp") -> tuple[nn.Module, callable]:
    """Factory returning (model, sample_input_fn).

    Args:
        name: "mlp" or "resnet18"

    Returns:
        (model, sample_input_fn) where sample_input_fn(device, batch_size) -> Tensor
    """
    if name == "mlp":
        return SimpleMLP(), _mlp_sample_input

    if name == "resnet18":
        try:
            from torchvision.models import resnet18
        except ImportError:
            raise ImportError(
                "torchvision is required for resnet18. "
                "Install with: pip install cuda-ipc-poc[vision]"
            )
        return resnet18(weights=None), _resnet_sample_input

    raise ValueError(f"Unknown model: {name!r}. Choose 'mlp' or 'resnet18'.")
