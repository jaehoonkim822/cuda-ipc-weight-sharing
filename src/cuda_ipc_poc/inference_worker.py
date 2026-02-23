"""Inference Worker — short-lived process that loads shared weights and runs inference.

Lifecycle:
  1. Connect to Weight Manager(s) via HandleClient
  2. Decode IPC handles and reconstruct shared GPU tensors
  3. Single endpoint: inject via load_state_dict(assign=True) for zero-copy
     Multiple endpoints (TP): collect shards from each WM, build TPSimpleMLP
  4. Run inference
  5. Release references and clean up
"""

import logging
import os

import torch
import torch.nn as nn

from .config import CUDA_ALLOC_CONF, DEVICE, ZMQ_ENDPOINT
from .handle_codec import decode_handles
from .ipc_channel import HandleClient
from .model import get_model
from .tensor_parallel import TPSimpleMLP

log = logging.getLogger(__name__)

_BUFFER_PREFIX = "__buffer__"


class InferenceWorker:
    def __init__(
        self,
        model_name: str = "mlp",
        device: str = DEVICE,
        endpoint: str | list[str] = ZMQ_ENDPOINT,
    ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF

        self._model_name = model_name
        self._device = device
        # Normalize to list for uniform handling
        if isinstance(endpoint, str):
            self._endpoints = [endpoint]
        else:
            self._endpoints = list(endpoint)
        self._model: nn.Module | None = None
        self._sample_input_fn = None
        # Keep references to shared storages to prevent premature GC
        self._shared_storages: list[torch.UntypedStorage] = []

    @property
    def _endpoint(self) -> str:
        """Return the single endpoint (for backward compat with scripts)."""
        return self._endpoints[0]

    def connect_and_load(self) -> None:
        """Fetch IPC handles from Weight Manager(s) and reconstruct model."""
        # CUDA context must be initialized before opening IPC handles.
        torch.cuda._lazy_init()

        # Fetch handles from all endpoints
        all_decoded = []
        for ep in self._endpoints:
            client = HandleClient(ep)
            raw = client.fetch_handles()
            decoded = decode_handles(raw)
            all_decoded.append(decoded)

        # Check if this is TP mode (first decoded has "tp_rank" key)
        if len(all_decoded) > 1 or "tp_rank" in all_decoded[0]:
            self._connect_tp(all_decoded)
        else:
            self._connect_single(all_decoded[0])

    def _connect_single(self, handles: dict) -> None:
        """Single-endpoint mode: reconstruct model from flat handle dict."""
        log.info("Received %d tensor handles", len(handles))

        state_dict = {}
        for name, info in handles.items():
            tensor = self._reconstruct_tensor(info, self._device)
            key = name[len(_BUFFER_PREFIX):] if name.startswith(_BUFFER_PREFIX) else name
            state_dict[key] = tensor
            log.debug("Reconstructed tensor '%s': shape=%s, dtype=%s", key, tensor.shape, tensor.dtype)

        model, self._sample_input_fn = get_model(self._model_name)
        self._model = model.to(self._device)
        # nn.Module method that sets evaluation mode
        self._model.eval()  # noqa: S307
        self._model.load_state_dict(state_dict, strict=True, assign=True)
        log.info("Model loaded with shared weights (zero-copy)")

    def _connect_tp(self, all_decoded: list[dict]) -> None:
        """Multi-endpoint TP mode: collect shards from each WM and build TPSimpleMLP."""
        rank_shards: dict[int, dict[str, torch.Tensor]] = {}

        for decoded in all_decoded:
            tp_rank = decoded["tp_rank"]
            tp_world_size = decoded["tp_world_size"]
            handles = decoded["handles"]

            # Determine the device for this rank's tensors from the IPC metadata
            # IPC handle metadata[0] is the device index
            first_info = next(iter(handles.values()))
            device_idx = first_info["metadata"][0]
            device = f"cuda:{device_idx}"

            # Initialize CUDA context on this device if needed
            torch.cuda.set_device(device)
            torch.cuda._lazy_init()

            shard = {}
            for name, info in handles.items():
                tensor = self._reconstruct_tensor(info, device)
                shard[name] = tensor
                log.debug("TP rank %d: reconstructed '%s': shape=%s", tp_rank, name, tensor.shape)

            rank_shards[tp_rank] = shard
            log.info("TP rank %d/%d: received %d shard handles", tp_rank, tp_world_size, len(handles))

        self._model = TPSimpleMLP(rank_shards, tp_world_size)
        _, self._sample_input_fn = get_model(self._model_name)
        log.info("TP model built with %d ranks", tp_world_size)

    def _reconstruct_tensor(self, info: dict, device: str) -> torch.Tensor:
        """Reconstruct a tensor from IPC handle metadata."""
        metadata = info["metadata"]
        storage = torch.UntypedStorage._new_shared_cuda(*metadata)
        self._shared_storages.append(storage)

        dtype = info["dtype"]
        size = info["size"]
        stride = info["stride"]
        storage_offset = info["storage_offset"]

        tensor = torch.empty([], dtype=dtype, device=device)
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
