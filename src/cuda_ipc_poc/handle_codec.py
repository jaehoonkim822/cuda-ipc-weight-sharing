"""IPC handle serialization/deserialization.

Serializes the per-tensor metadata needed to reconstruct shared GPU tensors
in another process. Uses pickle for simplicity — this is trusted same-machine
IPC only. Production use should replace with MessagePack or Protocol Buffers.

SECURITY NOTE: pickle is used intentionally here. CUDA IPC handles are only
valid between processes on the same physical machine with the same GPU, so
the attack surface is limited to local processes that already share hardware.

Handle dict structure per parameter:
{
    "fc1.weight": {
        "metadata": (device, handle_bytes, size, offset, ref_handle, ref_offset,
                     event_handle, event_sync),
        "size": [256, 784],
        "stride": [784, 1],
        "dtype": "torch.float32",
        "storage_offset": 0,
    },
    ...
}
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


def encode_handles(handles_dict: dict) -> bytes:
    """Serialize handle dict to bytes for socket transmission.

    Converts torch.dtype objects to strings for safe serialization.
    """
    serializable = {}
    for name, info in handles_dict.items():
        serializable[name] = {
            "metadata": info["metadata"],
            "size": list(info["size"]),
            "stride": list(info["stride"]),
            "dtype": _dtype_to_str(info["dtype"]),
            "storage_offset": info["storage_offset"],
        }
    return pickle.dumps(serializable, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301


def decode_handles(data: bytes) -> dict:
    """Deserialize bytes back to handle dict.

    Restores torch.dtype from string representation.
    """
    raw = pickle.loads(data)  # noqa: S301 — trusted same-machine IPC
    handles = {}
    for name, info in raw.items():
        handles[name] = {
            "metadata": info["metadata"],
            "size": info["size"],
            "stride": info["stride"],
            "dtype": _str_to_dtype(info["dtype"]),
            "storage_offset": info["storage_offset"],
        }
    return handles
