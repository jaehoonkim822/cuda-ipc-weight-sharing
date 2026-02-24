"""Declarative Tensor Parallel sharding handlers.

Each handler defines COLUMN_WISE_PARAMS and ROW_WISE_PARAMS patterns.
process_state_dict() applies pattern-matching to split parameters across ranks,
replicating any unmatched parameters.

Usage:
    handler = SimpleMlpTPHandler()
    shard = handler.process_state_dict(model.state_dict(), world_size=2, rank=0)
"""

from __future__ import annotations

import fnmatch

import torch


class BaseTensorParallelHandler:
    """Base class for declarative TP sharding.

    Subclasses define which parameters to split column-wise (dim=0)
    or row-wise (dim=1). Parameters not matching any pattern are replicated.
    """

    COLUMN_WISE_PARAMS: list[str] = []
    ROW_WISE_PARAMS: list[str] = []
    HAS_BUILTIN_TP_FORWARD: bool = False

    def _matches(self, name: str, patterns: list[str]) -> bool:
        """Check if parameter name matches any pattern (supports fnmatch globs)."""
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    def process_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        world_size: int,
        rank: int,
    ) -> dict[str, torch.Tensor]:
        """Split state_dict according to declared patterns.

        Column-wise params are chunked along dim=0 (output dimension).
        Row-wise params are chunked along dim=1 (input dimension).
        Unmatched params are replicated (returned as-is).
        """
        result: dict[str, torch.Tensor] = {}
        for name, param in state_dict.items():
            if self._matches(name, self.COLUMN_WISE_PARAMS):
                result[name] = param.chunk(world_size, dim=0)[rank].contiguous()
            elif self._matches(name, self.ROW_WISE_PARAMS):
                result[name] = param.chunk(world_size, dim=1)[rank].contiguous()
            else:
                result[name] = param
        return result


class SimpleMlpTPHandler(BaseTensorParallelHandler):
    """TP handler for SimpleMLP (784 -> 256 -> 10).

    fc1: column-parallel (split output dim)
    fc2.weight: row-parallel (split input dim)
    fc2.bias: rank 0 only (added after all-reduce)
    """

    COLUMN_WISE_PARAMS = ["fc1.weight", "fc1.bias"]
    ROW_WISE_PARAMS = ["fc2.weight"]

    def process_state_dict(self, state_dict, world_size, rank):
        result = super().process_state_dict(state_dict, world_size, rank)
        if rank != 0:
            result.pop("fc2.bias", None)
        return result
