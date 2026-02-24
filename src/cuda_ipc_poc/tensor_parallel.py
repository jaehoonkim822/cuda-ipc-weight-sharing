"""Tensor Parallelism utilities for Multi-WM architecture.

Sharding strategy for SimpleMLP (784 -> 256 -> 10):
  fc1 (Column-parallel): weight [256/N, 784], bias [256/N]
  fc2 (Row-parallel):    weight [10, 256/N], bias on rank 0 only

Each WM process calls shard_model() with its own rank to get its shard.
The Worker collects all shards and uses TPSimpleMLP for TP-aware forward.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def shard_model(
    model: nn.Module,
    world_size: int,
    rank: int,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Return one rank's shard of the model weights.

    Each WM calls this with its own rank to get the tensors it will serve.
    """
    state = model.state_dict()
    shard = {}

    # fc1: column-parallel — split along output dim (dim=0)
    fc1_w_chunks = state["fc1.weight"].chunk(world_size, dim=0)
    shard["fc1.weight"] = fc1_w_chunks[rank].to(device).contiguous()

    fc1_b_chunks = state["fc1.bias"].chunk(world_size, dim=0)
    shard["fc1.bias"] = fc1_b_chunks[rank].to(device).contiguous()

    # fc2: row-parallel — split along input dim (dim=1)
    fc2_w_chunks = state["fc2.weight"].chunk(world_size, dim=1)
    shard["fc2.weight"] = fc2_w_chunks[rank].to(device).contiguous()

    # fc2.bias: only rank 0 owns it (to avoid N-way duplication in all-reduce)
    if rank == 0:
        shard["fc2.bias"] = state["fc2.bias"].to(device).contiguous()

    return shard


class TPSimpleMLP(nn.Module):
    """TP-aware forward pass that combines shards from multiple ranks.

    Args:
        rank_shards: dict mapping rank -> dict of {name: Tensor}
        world_size: number of TP ranks
    """

    def __init__(self, rank_shards: dict[int, dict[str, torch.Tensor]], world_size: int):
        super().__init__()
        self._rank_shards = rank_shards
        self._world_size = world_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_0 = self._rank_shards[0]["fc1.weight"].device

        # Step 1: Column-parallel fc1 — each rank computes its slice
        partials_fc1 = []
        for rank in range(self._world_size):
            shard = self._rank_shards[rank]
            out = F.linear(x.to(shard["fc1.weight"].device), shard["fc1.weight"], shard["fc1.bias"])
            # Move to device_0 so we can concat across ranks
            partials_fc1.append(torch.relu(out).to(device_0))

        # Concatenate along feature dim: [batch, 256/N] * N -> [batch, 256]
        hidden = torch.cat(partials_fc1, dim=-1)

        # Step 2: Row-parallel fc2 — each rank computes partial output
        partial_outputs = []
        for rank in range(self._world_size):
            shard = self._rank_shards[rank]
            chunk_size = shard["fc2.weight"].shape[1]
            start = rank * chunk_size
            hidden_slice = hidden[..., start:start + chunk_size]
            out = F.linear(hidden_slice.to(shard["fc2.weight"].device), shard["fc2.weight"])
            partial_outputs.append(out)

        # Step 3: All-reduce (sum partial outputs on device_0) + bias
        result = torch.zeros_like(partial_outputs[0], device=device_0)
        for out in partial_outputs:
            result = result + out.to(device_0)

        if "fc2.bias" in self._rank_shards[0]:
            result = result + self._rank_shards[0]["fc2.bias"]

        return result


class DistributedTPForward(nn.Module):
    """Distributed TP forward using torch.distributed collectives.

    Each worker (rank) holds only its own shard and communicates via
    NCCL all_gather / all_reduce instead of local loops.

    Requires torch.distributed to be initialized before construction.
    """

    def __init__(
        self,
        shard: dict[str, torch.Tensor],
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self._rank = rank
        self._world_size = world_size

        self._fc1_weight = shard["fc1.weight"]
        self._fc1_bias = shard["fc1.bias"]
        self._fc2_weight = shard["fc2.weight"]

        # fc2.bias exists only on rank 0's shard.
        # Broadcast it so every rank can add it after all_reduce.
        if "fc2.bias" in shard:
            self._fc2_bias = shard["fc2.bias"].clone()
        else:
            self._fc2_bias = torch.zeros(
                self._fc2_weight.shape[0],
                dtype=self._fc2_weight.dtype,
                device=self._fc2_weight.device,
            )
        dist.broadcast(self._fc2_bias, src=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Column-parallel fc1: each rank computes its slice of the hidden layer
        local_fc1 = F.relu(F.linear(x, self._fc1_weight, self._fc1_bias))

        # all_gather: collect all ranks' hidden slices → full hidden [batch, 256]
        gathered = [torch.empty_like(local_fc1) for _ in range(self._world_size)]
        dist.all_gather(gathered, local_fc1)
        hidden = torch.cat(gathered, dim=-1)

        # Row-parallel fc2: each rank multiplies its slice of hidden
        chunk_size = self._fc2_weight.shape[1]
        start = self._rank * chunk_size
        hidden_slice = hidden[:, start : start + chunk_size]
        local_fc2 = F.linear(hidden_slice, self._fc2_weight)

        # all_reduce(SUM): combine partial fc2 outputs across ranks
        dist.all_reduce(local_fc2, op=dist.ReduceOp.SUM)

        return local_fc2 + self._fc2_bias
