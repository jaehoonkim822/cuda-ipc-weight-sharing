"""Unit tests for handle_codec — no GPU required."""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.handle_codec import (
    _dtype_to_str,
    _str_to_dtype,
    decode_handles,
    encode_handles,
)


def _make_fake_metadata():
    """Create a fake 8-tuple resembling _share_cuda_() output."""
    return (
        0,              # device index
        b"\x00" * 64,   # IPC handle bytes (cudaIpcMemHandle_t)
        1024,           # storage size
        0,              # storage offset
        b"\x00" * 64,   # ref counter handle
        0,              # ref counter offset
        b"\x00" * 64,   # event handle
        False,          # event sync required
    )


def _make_handles_dict():
    """Create a sample handles dict."""
    return {
        "fc1.weight": {
            "metadata": _make_fake_metadata(),
            "size": [256, 784],
            "stride": [784, 1],
            "dtype": torch.float32,
            "storage_offset": 0,
        },
        "fc1.bias": {
            "metadata": _make_fake_metadata(),
            "size": [256],
            "stride": [1],
            "dtype": torch.float32,
            "storage_offset": 0,
        },
        "__buffer__bn.running_mean": {
            "metadata": _make_fake_metadata(),
            "size": [64],
            "stride": [1],
            "dtype": torch.float64,
            "storage_offset": 0,
        },
    }


class TestDtypeConversion:
    def test_roundtrip_all_supported(self):
        dtypes = [
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.uint8,
        ]
        for dt in dtypes:
            assert _str_to_dtype(_dtype_to_str(dt)) is dt

    def test_unsupported_dtype(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _dtype_to_str(torch.complex64)

    def test_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            _str_to_dtype("imaginary128")


class TestEncodeDecodeRoundtrip:
    def test_basic_roundtrip(self):
        original = _make_handles_dict()
        encoded = encode_handles(original)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        decoded = decode_handles(encoded)

        assert set(decoded.keys()) == set(original.keys())
        for name in original:
            assert decoded[name]["size"] == original[name]["size"]
            assert decoded[name]["stride"] == original[name]["stride"]
            assert decoded[name]["dtype"] is original[name]["dtype"]
            assert decoded[name]["storage_offset"] == original[name]["storage_offset"]
            assert decoded[name]["metadata"] == original[name]["metadata"]

    def test_empty_dict(self):
        encoded = encode_handles({})
        decoded = decode_handles(encoded)
        assert decoded == {}

    def test_buffer_prefix_preserved(self):
        original = _make_handles_dict()
        decoded = decode_handles(encode_handles(original))
        assert "__buffer__bn.running_mean" in decoded

    def test_various_dtypes(self):
        handles = {}
        for i, dt in enumerate([torch.float16, torch.bfloat16, torch.int8, torch.bool]):
            handles[f"param_{i}"] = {
                "metadata": _make_fake_metadata(),
                "size": [10],
                "stride": [1],
                "dtype": dt,
                "storage_offset": 0,
            }
        decoded = decode_handles(encode_handles(handles))
        for i, dt in enumerate([torch.float16, torch.bfloat16, torch.int8, torch.bool]):
            assert decoded[f"param_{i}"]["dtype"] is dt

    def test_multidimensional_tensor_info(self):
        handles = {
            "conv.weight": {
                "metadata": _make_fake_metadata(),
                "size": [64, 3, 7, 7],
                "stride": [147, 49, 7, 1],
                "dtype": torch.float32,
                "storage_offset": 128,
            },
        }
        decoded = decode_handles(encode_handles(handles))
        assert decoded["conv.weight"]["size"] == [64, 3, 7, 7]
        assert decoded["conv.weight"]["stride"] == [147, 49, 7, 1]
        assert decoded["conv.weight"]["storage_offset"] == 128
