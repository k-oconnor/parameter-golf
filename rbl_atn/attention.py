"""RBL multi-head attention for sequence models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .casted_linear import CastedLinear
from .heads import make_head


class RBLMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_config: list[str],
        seq_len: int = 32,
    ) -> None:
        super().__init__()
        n_heads = len(head_config)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.head_config = head_config

        self.heads = nn.ModuleList(
            [make_head(h_type, d_model, self.d_head, seq_len) for h_type in head_config]
        )
        self.out_proj = CastedLinear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        causal: bool = False,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        head_outputs: list[Tensor] = []
        weight_dict: dict[int, Tensor] = {}

        for i, head in enumerate(self.heads):
            g = q_gain[i] if q_gain is not None else None
            out, weights = head(
                x,
                causal=causal,
                cos=cos,
                sin=sin,
                q_gain=g,
            )
            head_outputs.append(out)
            weight_dict[i] = weights

        concat = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(concat)
        return output, weight_dict

    def explain(self) -> str:
        lines = [f"RBLMultiHeadAttention  d_model={self.d_model}  n_heads={self.n_heads}"]
        for i, head in enumerate(self.heads):
            lines.append(f"  Head {i}: {head.explain()}")
        return "\n".join(lines)
