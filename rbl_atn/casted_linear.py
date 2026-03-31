"""CastedLinear — match train_gpt CastedLinear (fp32 weights, matmul in activations dtype)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)
