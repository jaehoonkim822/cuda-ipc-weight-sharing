"""Unit tests for tensor_parallel — CPU only, no GPU required."""

import multiprocessing
import os
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.model import SimpleMLP
from cuda_ipc_poc.tensor_parallel import DistributedTPForward, TPSimpleMLP, shard_model


@pytest.fixture
def model():
    """Create a deterministic SimpleMLP."""
    torch.manual_seed(0)
    m = SimpleMLP()
    # nn.Module method setting evaluation mode
    m.eval()  # noqa: S307
    return m


class TestShardModel:
    def test_shard_shapes_world_size_2(self, model):
        """fc1 column-parallel, fc2 row-parallel shapes for world_size=2."""
        shard_0 = shard_model(model, world_size=2, rank=0)
        shard_1 = shard_model(model, world_size=2, rank=1)

        # fc1 column-parallel: [256,784] -> [128,784] per rank
        assert shard_0["fc1.weight"].shape == (128, 784)
        assert shard_1["fc1.weight"].shape == (128, 784)
        assert shard_0["fc1.bias"].shape == (128,)
        assert shard_1["fc1.bias"].shape == (128,)

        # fc2 row-parallel: [10,256] -> [10,128] per rank
        assert shard_0["fc2.weight"].shape == (10, 128)
        assert shard_1["fc2.weight"].shape == (10, 128)

    def test_fc2_bias_only_rank_0(self, model):
        """fc2.bias should only exist on rank 0."""
        shard_0 = shard_model(model, world_size=2, rank=0)
        shard_1 = shard_model(model, world_size=2, rank=1)

        assert "fc2.bias" in shard_0
        assert "fc2.bias" not in shard_1
        assert shard_0["fc2.bias"].shape == (10,)

    def test_shards_reconstruct_original(self, model):
        """Concatenating shards should reconstruct the original weights."""
        state = model.state_dict()
        shard_0 = shard_model(model, world_size=2, rank=0)
        shard_1 = shard_model(model, world_size=2, rank=1)

        # fc1.weight: concat along dim=0
        reconstructed_fc1_w = torch.cat([shard_0["fc1.weight"], shard_1["fc1.weight"]], dim=0)
        assert torch.equal(reconstructed_fc1_w, state["fc1.weight"])

        # fc1.bias: concat along dim=0
        reconstructed_fc1_b = torch.cat([shard_0["fc1.bias"], shard_1["fc1.bias"]], dim=0)
        assert torch.equal(reconstructed_fc1_b, state["fc1.bias"])

        # fc2.weight: concat along dim=1
        reconstructed_fc2_w = torch.cat([shard_0["fc2.weight"], shard_1["fc2.weight"]], dim=1)
        assert torch.equal(reconstructed_fc2_w, state["fc2.weight"])

        # fc2.bias: rank 0 only
        assert torch.equal(shard_0["fc2.bias"], state["fc2.bias"])

    def test_shards_contiguous(self, model):
        """All shard tensors must be contiguous (required for CUDA IPC)."""
        for rank in range(2):
            shard = shard_model(model, world_size=2, rank=rank)
            for name, tensor in shard.items():
                assert tensor.is_contiguous(), f"rank {rank} {name} not contiguous"

    def test_world_size_4(self, model):
        """Verify shapes with world_size=4."""
        for rank in range(4):
            shard = shard_model(model, world_size=4, rank=rank)
            assert shard["fc1.weight"].shape == (64, 784)
            assert shard["fc1.bias"].shape == (64,)
            assert shard["fc2.weight"].shape == (10, 64)
            if rank == 0:
                assert "fc2.bias" in shard
            else:
                assert "fc2.bias" not in shard


class TestTPSimpleMLP:
    def test_tp_forward_matches_single_model(self, model):
        """TP forward pass should match single-model forward (CPU)."""
        # Build rank shards on CPU
        world_size = 2
        rank_shards = {
            rank: shard_model(model, world_size, rank)
            for rank in range(world_size)
        }

        tp_model = TPSimpleMLP(rank_shards, world_size)

        # Same input, compare outputs
        torch.manual_seed(42)
        x = torch.randn(4, 784)

        with torch.no_grad():
            ref_out = model(x)
            tp_out = tp_model(x)

        assert torch.allclose(ref_out, tp_out, atol=1e-5), (
            f"Max diff: {(ref_out - tp_out).abs().max().item()}"
        )

    def test_tp_forward_world_size_4(self, model):
        """TP forward with world_size=4 should also match."""
        world_size = 4
        rank_shards = {
            rank: shard_model(model, world_size, rank)
            for rank in range(world_size)
        }

        tp_model = TPSimpleMLP(rank_shards, world_size)

        torch.manual_seed(123)
        x = torch.randn(2, 784)

        with torch.no_grad():
            ref_out = model(x)
            tp_out = tp_model(x)

        assert torch.allclose(ref_out, tp_out, atol=1e-5), (
            f"Max diff: {(ref_out - tp_out).abs().max().item()}"
        )

    def test_tp_batch_consistency(self, model):
        """Different batch sizes should all produce matching outputs."""
        world_size = 2
        rank_shards = {
            rank: shard_model(model, world_size, rank)
            for rank in range(world_size)
        }
        tp_model = TPSimpleMLP(rank_shards, world_size)

        for batch_size in [1, 8, 32]:
            torch.manual_seed(7)
            x = torch.randn(batch_size, 784)
            with torch.no_grad():
                ref_out = model(x)
                tp_out = tp_model(x)
            assert torch.allclose(ref_out, tp_out, atol=1e-5), (
                f"Batch {batch_size} max diff: {(ref_out - tp_out).abs().max().item()}"
            )


# -- Distributed TP tests (gloo backend, CPU only) --

def _distributed_tp_worker(rank, world_size, state_dict_bytes, input_bytes, master_port, result_dict):
    """Entry point for each gloo process testing DistributedTPForward."""
    import torch.distributed as dist
    from cuda_ipc_poc.tensor_parallel import DistributedTPForward, shard_model
    from cuda_ipc_poc.model import SimpleMLP

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo")

    try:
        # Reconstruct model from state_dict to build shards
        model = SimpleMLP()
        model.load_state_dict(torch.load(
            __import__("io").BytesIO(state_dict_bytes), weights_only=True,
        ))
        shard = shard_model(model, world_size, rank)

        tp_fwd = DistributedTPForward(shard, rank, world_size)
        x = torch.load(__import__("io").BytesIO(input_bytes), weights_only=True)
        output = tp_fwd(x)

        result_dict[rank] = output.tolist()
    finally:
        dist.destroy_process_group()


def _run_distributed_tp_test(world_size, master_port):
    """Spawn gloo processes and compare against reference SimpleMLP output."""
    torch.manual_seed(0)
    model = SimpleMLP()
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(4, 784)

    with torch.no_grad():
        ref_output = model(x)

    # Serialize model state and input for transfer to child processes
    import io
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
            target=_distributed_tp_worker,
            args=(rank, world_size, state_bytes, input_bytes, master_port, result_dict),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=30)

    for rank, p in enumerate(procs):
        assert p.exitcode == 0, f"Rank {rank} exited with code {p.exitcode}"

    # Verify all ranks match reference
    for rank in range(world_size):
        rank_output = torch.tensor(result_dict[rank])
        assert torch.allclose(ref_output, rank_output, atol=1e-5), (
            f"Rank {rank} max diff: {(ref_output - rank_output).abs().max().item()}"
        )

    # Verify all ranks are identical (all_reduce guarantees this)
    rank0_output = torch.tensor(result_dict[0])
    for rank in range(1, world_size):
        rank_output = torch.tensor(result_dict[rank])
        assert torch.allclose(rank0_output, rank_output, atol=1e-6), (
            f"Rank 0 vs {rank} diff: {(rank0_output - rank_output).abs().max().item()}"
        )


class TestDistributedTPForward:
    def test_distributed_tp_world_size_2(self):
        """DistributedTPForward with gloo backend, world_size=2."""
        _run_distributed_tp_test(world_size=2, master_port=29601)

    def test_distributed_tp_world_size_4(self):
        """DistributedTPForward with gloo backend, world_size=4."""
        _run_distributed_tp_test(world_size=4, master_port=29602)
