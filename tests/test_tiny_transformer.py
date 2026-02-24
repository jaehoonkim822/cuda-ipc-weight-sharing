"""Unit tests for TinyTransformer model and TP sharding — CPU only."""

import io
import multiprocessing
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.models.tiny_llama import (
    TinyTransformer,
    TinyTransformerTPHandler,
    _tiny_transformer_sample_input,
)
from cuda_ipc_poc.inference_worker import _assign_sharded_state_dict
from cuda_ipc_poc.model_spec import ModelRegistry


class TestTinyTransformer:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        m = TinyTransformer(hidden=64, heads=4, layers=2, vocab=128)
        m.eval()
        return m

    def test_forward_shape(self, model):
        x = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8, 128)

    def test_forward_deterministic(self, model):
        x = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2)

    def test_registry_registration(self):
        # Importing the module triggers registration
        spec = ModelRegistry.get("tiny-transformer")
        assert spec.model_cls is TinyTransformer
        assert spec.tp_handler is not None
        assert spec.tp_handler.HAS_BUILTIN_TP_FORWARD is True

    def test_sample_input_fn(self):
        x = _tiny_transformer_sample_input("cpu", batch_size=3, seq_len=16)
        assert x.shape == (3, 16)
        assert x.dtype == torch.long


class TestTinyTransformerTPHandler:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        m = TinyTransformer(hidden=64, heads=4, layers=2, vocab=128)
        m.eval()
        return m

    def test_shard_shapes_world_size_2(self, model):
        handler = TinyTransformerTPHandler()
        state = model.state_dict()
        shard = handler.process_state_dict(state, world_size=2, rank=0)

        # Column-parallel: q/k/v_proj [64, 64] → [32, 64], gate/up [256, 64] → [128, 64]
        assert shard["layers.0.self_attn.q_proj.weight"].shape == (32, 64)
        assert shard["layers.0.self_attn.k_proj.weight"].shape == (32, 64)
        assert shard["layers.0.self_attn.v_proj.weight"].shape == (32, 64)
        assert shard["layers.0.mlp.gate_proj.weight"].shape == (128, 64)
        assert shard["layers.0.mlp.up_proj.weight"].shape == (128, 64)

        # Row-parallel: o_proj [64, 64] → [64, 32], down [64, 256] → [64, 128]
        assert shard["layers.0.self_attn.o_proj.weight"].shape == (64, 32)
        assert shard["layers.0.mlp.down_proj.weight"].shape == (64, 128)

        # Replicated: norms, embedding, lm_head
        assert shard["layers.0.input_layernorm.weight"].shape == (64,)
        assert shard["embed.weight"].shape == (128, 64)
        assert shard["lm_head.weight"].shape == (128, 64)

    def test_shard_shapes_layer_1(self, model):
        handler = TinyTransformerTPHandler()
        state = model.state_dict()
        shard = handler.process_state_dict(state, world_size=2, rank=1)

        # Layer 1 should have same shapes as layer 0
        assert shard["layers.1.self_attn.q_proj.weight"].shape == (32, 64)
        assert shard["layers.1.mlp.down_proj.weight"].shape == (64, 128)

    def test_shards_reconstruct_original(self, model):
        handler = TinyTransformerTPHandler()
        state = model.state_dict()
        r0 = handler.process_state_dict(state, world_size=2, rank=0)
        r1 = handler.process_state_dict(state, world_size=2, rank=1)

        # Column-parallel: concat along dim=0
        q_reconstructed = torch.cat([
            r0["layers.0.self_attn.q_proj.weight"],
            r1["layers.0.self_attn.q_proj.weight"],
        ], dim=0)
        assert torch.equal(q_reconstructed, state["layers.0.self_attn.q_proj.weight"])

        # Row-parallel: concat along dim=1
        o_reconstructed = torch.cat([
            r0["layers.0.self_attn.o_proj.weight"],
            r1["layers.0.self_attn.o_proj.weight"],
        ], dim=1)
        assert torch.equal(o_reconstructed, state["layers.0.self_attn.o_proj.weight"])

    def test_contiguity(self, model):
        handler = TinyTransformerTPHandler()
        state = model.state_dict()
        shard = handler.process_state_dict(state, world_size=2, rank=0)
        for name, tensor in shard.items():
            assert tensor.is_contiguous(), f"{name} not contiguous"

    def test_world_size_4(self, model):
        handler = TinyTransformerTPHandler()
        state = model.state_dict()
        for rank in range(4):
            shard = handler.process_state_dict(state, world_size=4, rank=rank)
            # q_proj: [64, 64] → [16, 64]
            assert shard["layers.0.self_attn.q_proj.weight"].shape == (16, 64)
            # gate_proj: [256, 64] → [64, 64]
            assert shard["layers.0.mlp.gate_proj.weight"].shape == (64, 64)


# -- Distributed TP test (gloo, CPU) --

def _distributed_tiny_transformer_worker(rank, world_size, state_bytes, input_bytes, master_port, result_dict):
    """Worker process for distributed TP test."""
    import torch.distributed as dist
    from cuda_ipc_poc.models.tiny_llama import TinyTransformer, TinyTransformerTPHandler
    from cuda_ipc_poc.inference_worker import _assign_sharded_state_dict

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo")

    try:
        full_state = torch.load(io.BytesIO(state_bytes), weights_only=True)
        handler = TinyTransformerTPHandler()
        shard = handler.process_state_dict(full_state, world_size, rank)

        model = TinyTransformer(hidden=64, heads=4, layers=2, vocab=128, tp_world_size=world_size)
        model.eval()
        _assign_sharded_state_dict(model, shard)

        x = torch.load(io.BytesIO(input_bytes), weights_only=True)
        with torch.no_grad():
            output = model(x)

        result_dict[rank] = output.tolist()
    finally:
        dist.destroy_process_group()


class TestTinyTransformerDistributedTP:
    def _run_test(self, world_size, master_port):
        torch.manual_seed(0)
        model = TinyTransformer(hidden=64, heads=4, layers=2, vocab=128)
        model.eval()

        x = torch.randint(0, 128, (2, 8))

        with torch.no_grad():
            ref_output = model(x)

        state_buf = io.BytesIO()
        torch.save(model.state_dict(), state_buf)
        state_bytes = state_buf.getvalue()

        input_buf = io.BytesIO()
        torch.save(x, input_buf)
        input_bytes = input_buf.getvalue()

        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_distributed_tiny_transformer_worker,
                args=(rank, world_size, state_bytes, input_bytes, master_port, result_dict),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=30)

        for rank, p in enumerate(procs):
            assert p.exitcode == 0, f"Rank {rank} exited with code {p.exitcode}"

        # All ranks should produce the same output (due to all_reduce)
        for rank in range(world_size):
            rank_output = torch.tensor(result_dict[rank])
            assert torch.allclose(ref_output, rank_output, atol=1e-4), (
                f"Rank {rank} max diff: {(ref_output - rank_output).abs().max().item()}"
            )

    def test_distributed_tp_world_size_2(self):
        self._run_test(world_size=2, master_port=29701)

    def test_distributed_tp_world_size_4(self):
        self._run_test(world_size=4, master_port=29702)
