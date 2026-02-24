"""Weight Manager — long-lived process that loads model weights and serves IPC handles.

Lifecycle:
  1. Set PYTORCH_CUDA_ALLOC_CONF to disable expandable_segments
  2. Load model to GPU, freeze parameters
  3. Export IPC handles for every parameter and buffer
  4. Serve handles over ZMQ REP socket
  5. Periodically call ipc_collect() to reclaim memory from dead workers
"""

import logging
import os
import signal
import time

import torch
import torch.nn as nn

from .config import CUDA_ALLOC_CONF, DEVICE, IPC_COLLECT_INTERVAL, ZMQ_ENDPOINT
from .handle_codec import encode_handles
from .ipc_channel import HandleServer
from .model_spec import ModelRegistry

log = logging.getLogger(__name__)


class WeightManager:
    def __init__(
        self,
        model_name: str = "mlp",
        device: str = DEVICE,
        endpoint: str = ZMQ_ENDPOINT,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        model_path: str | None = None,
    ):
        # Must be set before any CUDA operation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
        log.info("PYTORCH_CUDA_ALLOC_CONF=%s", CUDA_ALLOC_CONF)

        self._model_name = model_name
        self._device = device
        self._endpoint = endpoint
        self._tp_rank = tp_rank
        self._tp_world_size = tp_world_size
        self._model_path = model_path
        self._model: nn.Module | None = None
        self._sample_input_fn = None
        self._server = HandleServer(endpoint)
        self._running = True

    def load_model(self) -> None:
        """Load model to GPU and export IPC handles."""
        # Ensure model registrations are loaded
        import cuda_ipc_poc.model  # noqa: F401

        spec = ModelRegistry.get(self._model_name)
        self._sample_input_fn = spec.sample_input_fn

        if spec.weight_loader and self._model_path:
            state_dict = spec.weight_loader.load(self._model_path, self._device)
            model = spec.model_factory()
            model.load_state_dict(state_dict)
        else:
            model = spec.model_factory()

        self._model = model.to(self._device)
        # nn.Module method that sets evaluation mode (disables dropout etc.)
        self._model.eval()  # noqa: S307
        self._model.requires_grad_(False)
        log.info(
            "Model '%s' loaded to %s (%d parameters, %d buffers)",
            self._model_name,
            self._device,
            sum(1 for _ in self._model.parameters()),
            sum(1 for _ in self._model.buffers()),
        )

        if self._tp_world_size > 1:
            if spec.tp_handler is None:
                raise ValueError(
                    f"Model {self._model_name!r} does not support tensor parallelism "
                    "(no tp_handler in ModelSpec)"
                )
            full_state = self._model.state_dict()
            shard = spec.tp_handler.process_state_dict(
                full_state, self._tp_world_size, self._tp_rank,
            )
            # Move shard tensors to device and ensure contiguity
            shard = {k: v.to(self._device).contiguous() for k, v in shard.items()}
            handles = {}
            for name, tensor in shard.items():
                handles[name] = self._tensor_to_handle(tensor)
            # Keep shard tensors alive by storing references
            self._shard_tensors = shard
            encoded = encode_handles(handles, tp_rank=self._tp_rank, tp_world_size=self._tp_world_size)
            log.info(
                "TP rank %d/%d: exported %d shard handles (%d bytes)",
                self._tp_rank, self._tp_world_size, len(handles), len(encoded),
            )
        else:
            handles = self._export_handles()
            encoded = encode_handles(handles)
            log.info("Exported %d tensor handles (%d bytes)", len(handles), len(encoded))

        self._server.set_handles(encoded)

    def _export_handles(self) -> dict:
        """Extract IPC handles for all parameters and buffers."""
        handles = {}

        for name, param in self._model.named_parameters():
            handles[name] = self._tensor_to_handle(param.data)

        # Prefix buffers with __buffer__ to distinguish from parameters
        for name, buf in self._model.named_buffers():
            handles[f"__buffer__{name}"] = self._tensor_to_handle(buf)

        return handles

    @staticmethod
    def _tensor_to_handle(tensor: torch.Tensor) -> dict:
        """Convert a GPU tensor to its IPC handle metadata."""
        storage = tensor.untyped_storage()
        metadata = storage._share_cuda_()
        return {
            "metadata": metadata,
            "size": list(tensor.size()),
            "stride": list(tensor.stride()),
            "dtype": tensor.dtype,
            "storage_offset": tensor.storage_offset(),
        }

    def serve(self) -> None:
        """Start serving handles and run ipc_collect loop until interrupted."""
        self._server.serve_in_background()
        log.info("Weight Manager serving on %s", self._endpoint)

        # Graceful shutdown on SIGINT/SIGTERM
        def _shutdown(signum, frame):
            sig_name = signal.Signals(signum).name
            log.info("Received %s, shutting down...", sig_name)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while self._running:
            time.sleep(IPC_COLLECT_INTERVAL)
            if self._running:
                torch.cuda.ipc_collect()
                log.debug("ipc_collect() completed")

        self._server.stop()
        log.info("Weight Manager stopped")

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def sample_input_fn(self):
        return self._sample_input_fn
