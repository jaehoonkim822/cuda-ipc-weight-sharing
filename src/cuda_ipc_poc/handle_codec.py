"""IPC handle serialization/deserialization.

Serializes the per-tensor metadata needed to reconstruct shared GPU tensors
in another process. Uses pickle for simplicity — this is trusted same-machine
IPC only. Production use should replace with MessagePack or Protocol Buffers.

SECURITY NOTE: pickle is used intentionally here. CUDA IPC handles are only
valid between processes on the same physical machine with the same GPU, so
the attack surface is limited to local processes that already share hardware.

Supports two formats:
  - Flat dict (legacy single-GPU): {name: handle_info, ...}
  - TP dict (multi-GPU): {"tp_rank": int, "tp_world_size": int, "handles": {name: handle_info, ...}}
"""

import pickle  # noqa: S403 — intentional, trusted same-machine IPC only

import torch

# torch.dtype <-> string mapping
_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.uint8: "uint8",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_to_str(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_STR[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype for IPC: {dtype}")


def _str_to_dtype(s: str) -> torch.dtype:
    try:
        return _STR_TO_DTYPE[s]
    except KeyError:
        raise ValueError(f"Unknown dtype string: {s!r}")


def _encode_single_handle(info: dict) -> dict:
    """Convert a single handle info dict to serializable form."""
    return {
        "metadata": info["metadata"],
        "size": list(info["size"]),
        "stride": list(info["stride"]),
        "dtype": _dtype_to_str(info["dtype"]),
        "storage_offset": info["storage_offset"],
    }


def _decode_single_handle(info: dict) -> dict:
    """Restore a single handle info dict from serialized form."""
    return {
        "metadata": info["metadata"],
        "size": info["size"],
        "stride": info["stride"],
        "dtype": _str_to_dtype(info["dtype"]),
        "storage_offset": info["storage_offset"],
    }


def encode_handles(handles_dict: dict, tp_rank: int | None = None, tp_world_size: int | None = None) -> bytes:
    """Serialize handle dict to bytes for socket transmission.

    If tp_rank is provided, wraps in TP envelope format.
    Pickle is used intentionally for trusted same-machine IPC (see module docstring).
    """
    serializable_handles = {
        name: _encode_single_handle(info)
        for name, info in handles_dict.items()
    }

    if tp_rank is not None:
        payload = {
            "tp_rank": tp_rank,
            "tp_world_size": tp_world_size,
            "handles": serializable_handles,
        }
    else:
        payload = serializable_handles

    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301


def decode_handles(data: bytes) -> dict:
    """Deserialize bytes back to handle dict.

    Auto-detects TP envelope format vs legacy flat format.
    Returns the full dict (including tp_rank/tp_world_size if present).
    Pickle is used intentionally for trusted same-machine IPC (see module docstring).
    """
    raw = pickle.loads(data)  # noqa: S301

    # Detect TP format by presence of "tp_rank" key
    if "tp_rank" in raw:
        handles = {
            name: _decode_single_handle(info)
            for name, info in raw["handles"].items()
        }
        return {
            "tp_rank": raw["tp_rank"],
            "tp_world_size": raw["tp_world_size"],
            "handles": handles,
        }

    # Legacy flat format
    return {
        name: _decode_single_handle(info)
        for name, info in raw.items()
    }
