"""Shared constants for CUDA IPC PoC.

All values can be overridden via environment variables.
"""

import os

ZMQ_ENDPOINT = os.environ.get("CUDA_IPC_ZMQ_ENDPOINT", "ipc:///tmp/cuda_ipc_wm.zmq")

DEVICE = os.environ.get("CUDA_IPC_DEVICE", "cuda:0")

# CRITICAL: expandable_segments must be disabled for CUDA IPC compatibility.
# PyTorch 2.2+ may enable it by default, which uses cudaMemCreate/cudaMemMap
# instead of cudaMalloc, breaking cudaIpcGetMemHandle.
CUDA_ALLOC_CONF = "expandable_segments:False"

TP_WORLD_SIZE = int(os.environ.get("CUDA_IPC_TP_WORLD_SIZE", "1"))

# Interval (seconds) for calling torch.cuda.ipc_collect() in the Weight Manager.
# This reclaims GPU memory from terminated worker processes.
IPC_COLLECT_INTERVAL = 5
