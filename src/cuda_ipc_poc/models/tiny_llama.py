"""TinyTransformer — minimal LLaMA-style model for CUDA IPC TP validation.

Architecture:
  - Embedding (vocab → hidden)
  - N TransformerBlocks: RMSNorm → MultiHeadAttention → RMSNorm → MLP (gate/up/down)
  - Final RMSNorm → Linear head (hidden → vocab)

TP strategy (same as LLaMA):
  - Column-parallel: q_proj, k_proj, v_proj, gate_proj, up_proj
  - Row-parallel: o_proj, down_proj

When TP is active, the model's forward() uses dist.all_reduce after each
row-parallel layer, so no external TP wrapper is needed.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..model_spec import ModelRegistry, ModelSpec
from ..tp_handler import BaseTensorParallelHandler


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, tp_world_size: int = 1):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.tp_world_size = tp_world_size

        # Column-parallel: each TP rank holds heads/N heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        # Row-parallel: output projection
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Infer local head count from the projected dimension
        local_heads = q.shape[-1] // self.head_dim

        q = q.view(B, T, local_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, local_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, local_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, local_heads * self.head_dim)
        out = self.o_proj(out)

        if self.tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, tp_world_size: int = 1):
        super().__init__()
        self.tp_world_size = tp_world_size
        # SwiGLU-style: gate_proj and up_proj are column-parallel
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        # down_proj is row-parallel
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        if self.tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int, tp_world_size: int = 1):
        super().__init__()
        self.self_attn = Attention(hidden, heads, tp_world_size)
        self.mlp = MLP(hidden, intermediate, tp_world_size)
        self.input_layernorm = RMSNorm(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyTransformer(nn.Module):
    """Minimal LLaMA-style transformer with built-in TP support.

    Args:
        hidden: Hidden dimension (must be divisible by heads)
        heads: Number of attention heads
        layers: Number of transformer blocks
        vocab: Vocabulary size
        tp_world_size: Tensor parallel world size (1 = no TP)
    """

    def __init__(
        self,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        vocab: int = 128,
        tp_world_size: int = 1,
    ):
        super().__init__()
        intermediate = hidden * 4
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden, heads, intermediate, tp_world_size)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class TinyTransformerTPHandler(BaseTensorParallelHandler):
    """TP sharding for TinyTransformer following LLaMA parallelism strategy."""

    COLUMN_WISE_PARAMS = [
        "layers.*.self_attn.q_proj.weight",
        "layers.*.self_attn.k_proj.weight",
        "layers.*.self_attn.v_proj.weight",
        "layers.*.mlp.gate_proj.weight",
        "layers.*.mlp.up_proj.weight",
    ]
    ROW_WISE_PARAMS = [
        "layers.*.self_attn.o_proj.weight",
        "layers.*.mlp.down_proj.weight",
    ]
    HAS_BUILTIN_TP_FORWARD = True


def _tiny_transformer_sample_input(device: str, batch_size: int = 1, seq_len: int = 8) -> torch.Tensor:
    return torch.randint(0, 128, (batch_size, seq_len), device=device)


ModelRegistry.register("tiny-transformer", ModelSpec(
    model_cls=TinyTransformer,
    model_factory=lambda: TinyTransformer(hidden=64, heads=4, layers=2, vocab=128),
    sample_input_fn=_tiny_transformer_sample_input,
    tp_handler=TinyTransformerTPHandler(),
))
