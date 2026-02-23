"""Inference Worker — short-lived process that loads shared weights and runs inference.

Lifecycle:
  1. Connect to Weight Manager via HandleClient
  2. Decode IPC handles and reconstruct shared GPU tensors
  3. Inject into model via load_state_dict(assign=True) for zero-copy
  4. Run inference
  5. Release references and clean up
"""

import logging
import os

import torch
import torch.nn as nn

from .config import CUDA_ALLOC_CONF, DEVICE, SOCKET_PATH
from .handle_codec import decode_handles
from .ipc_channel import HandleClient
from .model import get_model

log = logging.getLogger(__name__)

_BUFFER_PREFIX = "__buffer__"


class InferenceWorker:
    def __init__(
        self,
        model_name: str = "mlp",
        device: str = DEVICE,
        socket_path: str = SOCKET_PATH,
    ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF

        self._model_name = model_name
        self._device = device
        self._socket_path = socket_path
        self._model: nn.Module | None = None
        self._sample_input_fn = None
        # Keep references to shared storages to prevent premature GC
        self._shared_storages: list[torch.UntypedStorage] = []

    def connect_and_load(self) -> None:
        """Fetch IPC handles from Weight Manager and reconstruct model."""
        # CUDA context must be initialized before opening IPC handles.
        # _new_shared_cuda calls cudaIpcOpenMemHandle which requires a valid context.
        torch.cuda._lazy_init()

        client = HandleClient(self._socket_path)
        raw = client.fetch_handles()
        handles = decode_handles(raw)
        log.info("Received %d tensor handles", len(handles))

        # Build state_dict from shared storages
        state_dict = {}
        for name, info in handles.items():
            tensor = self._reconstruct_tensor(info)

            # Strip __buffer__ prefix for buffers
            key = name[len(_BUFFER_PREFIX):] if name.startswith(_BUFFER_PREFIX) else name
            state_dict[key] = tensor
            log.debug(
                "Reconstructed tensor '%s': shape=%s, dtype=%s",
                key, tensor.shape, tensor.dtype,
            )

        # Create a fresh model shell and inject shared weights
        model, self._sample_input_fn = get_model(self._model_name)
        self._model = model.to(self._device)
        # nn.Module.eval() — sets evaluation mode (disables dropout etc.)
        self._model.eval()  # noqa: eval is nn.Module.eval(), not builtin
        # assign=True replaces parameter tensors instead of copying — zero-copy
        self._model.load_state_dict(state_dict, strict=True, assign=True)
        log.info("Model loaded with shared weights (zero-copy)")

    def _reconstruct_tensor(self, info: dict) -> torch.Tensor:
        """Reconstruct a tensor from IPC handle metadata."""
        metadata = info["metadata"]
        # Open the shared CUDA storage from the IPC handle
        storage = torch.UntypedStorage._new_shared_cuda(*metadata)
        self._shared_storages.append(storage)

        # Create a typed tensor view with correct shape/stride
        dtype = info["dtype"]
        size = info["size"]
        stride = info["stride"]
        storage_offset = info["storage_offset"]

        tensor = torch.empty([], dtype=dtype, device=self._device)
        tensor.set_(storage, storage_offset, size, stride)
        return tensor

    @torch.no_grad()
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass on the shared model."""
        return self._model(input_tensor)

    def get_data_ptrs(self) -> dict[str, int]:
        """Return data_ptr() for each parameter — used to verify zero-copy sharing."""
        ptrs = {}
        for name, param in self._model.named_parameters():
            ptrs[name] = param.data_ptr()
        for name, buf in self._model.named_buffers():
            ptrs[f"{_BUFFER_PREFIX}{name}"] = buf.data_ptr()
        return ptrs

    def cleanup(self) -> None:
        """Release all references to shared GPU memory."""
        self._model = None
        self._shared_storages.clear()
        torch.cuda.empty_cache()
        log.info("Worker cleaned up shared references")
