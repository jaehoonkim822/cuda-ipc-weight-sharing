"""Tensor Parallelism utilities for Multi-WM architecture.

Sharding strategy for SimpleMLP (784 -> 256 -> 10):
  fc1 (Column-parallel): weight [256/N, 784], bias [256/N]
  fc2 (Row-parallel):    weight [10, 256/N], bias on rank 0 only

Each WM process calls shard_model() with its own rank to get its shard.
The Worker collects all shards and uses TPSimpleMLP for TP-aware forward.
"""

import torch
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
