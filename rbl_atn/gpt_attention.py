"""RBL-ATN block compatible with train_gpt Block (causal LM, RoPE, CastedLinear)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .attention import RBLMultiHeadAttention
from .rope import Rotary


class RBLCausalSelfAttention(nn.Module):
    """Causal self-attention using typed RBL heads (no GQA — use NUM_KV_HEADS=NUM_HEADS)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        seq_len: int,
        head_config: list[str],
    ) -> None:
        super().__init__()
        if num_kv_heads != num_heads:
            raise ValueError(
                "RBL attention does not use GQA; set NUM_KV_HEADS equal to NUM_HEADS "
                f"(got num_kv_heads={num_kv_heads}, num_heads={num_heads})"
            )
        if len(head_config) != num_heads:
            raise ValueError(
                f"RBL_HEAD_CONFIG must list exactly NUM_HEADS ({num_heads}) types; got {len(head_config)}"
            )
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        head_dim = dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rbl = RBLMultiHeadAttention(dim, head_config, seq_len=seq_len)
        self.rbl.out_proj._zero_init = True  # noqa: SLF001
        self.rotary = Rotary(head_dim, base=rope_base)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        cos, sin = self.rotary(seqlen, x.device, x.dtype)
        y, _ = self.rbl(x, causal=True, cos=cos, sin=sin, q_gain=self.q_gain)
        return y
