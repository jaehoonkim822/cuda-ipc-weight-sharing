"""Weight Manager — long-lived process that loads model weights and serves IPC handles.

Lifecycle:
  1. Set PYTORCH_CUDA_ALLOC_CONF to disable expandable_segments
  2. Load model to GPU, freeze parameters
  3. Export IPC handles for every parameter and buffer
  4. Serve handles over Unix Domain Socket
  5. Periodically call ipc_collect() to reclaim memory from dead workers
"""

import logging
import os
import signal
import time

import torch
import torch.nn as nn

from .config import CUDA_ALLOC_CONF, DEVICE, IPC_COLLECT_INTERVAL, SOCKET_PATH
from .handle_codec import encode_handles
from .ipc_channel import HandleServer
from .model import get_model

log = logging.getLogger(__name__)


class WeightManager:
    def __init__(
        self,
        model_name: str = "mlp",
        device: str = DEVICE,
        socket_path: str = SOCKET_PATH,
    ):
        # Must be set before any CUDA operation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
        log.info("PYTORCH_CUDA_ALLOC_CONF=%s", CUDA_ALLOC_CONF)

        self._model_name = model_name
        self._device = device
        self._socket_path = socket_path
        self._model: nn.Module | None = None
        self._sample_input_fn = None
        self._server = HandleServer(socket_path)
        self._running = True

    def load_model(self) -> None:
        """Load model to GPU and export IPC handles."""
        model, self._sample_input_fn = get_model(self._model_name)
        # .eval() here is nn.Module.eval(), setting evaluation mode — not Python eval()
        self._model = model.to(self._device)
        self._model.eval()
        self._model.requires_grad_(False)
        log.info(
            "Model '%s' loaded to %s (%d parameters, %d buffers)",
            self._model_name,
            self._device,
            sum(1 for _ in self._model.parameters()),
            sum(1 for _ in self._model.buffers()),
        )

        handles = self._export_handles()
        encoded = encode_handles(handles)
        self._server.set_handles(encoded)
        log.info("Exported %d tensor handles (%d bytes)", len(handles), len(encoded))

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
        log.info("Weight Manager serving on %s", self._socket_path)

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
