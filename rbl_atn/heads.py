"""Typed RBL attention heads with optional RoPE + per-head q gain (GPT integration)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .casted_linear import CastedLinear
from .kernels import AlwaysKernel, EventuallyKernel, NextKernel, UntilKernel
from .rope import apply_rotary_emb


class RBLHead(nn.Module):
    logical_type: str = "BASE"

    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.scale = math.sqrt(d_head)

        self.W_q = CastedLinear(d_model, d_head, bias=False)
        self.W_k = CastedLinear(d_model, d_head, bias=False)
        self.W_v = CastedLinear(d_model, d_head, bias=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        return q, k, v

    @staticmethod
    def _apply_rotary_qk(
        q: Tensor, k: Tensor, cos: Tensor | None, sin: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if cos is None or sin is None:
            return q, k
        q = apply_rotary_emb(q.unsqueeze(1), cos, sin).squeeze(1)
        k = apply_rotary_emb(k.unsqueeze(1), cos, sin).squeeze(1)
        return q, k

    @staticmethod
    def _apply_causal(weights: Tensor) -> Tensor:
        T = weights.size(1)
        mask = torch.tril(torch.ones(T, T, device=weights.device, dtype=weights.dtype))
        w = weights * mask
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        return w

    def forward(
        self,
        x: Tensor,
        causal: bool = False,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def explain(self) -> str:
        raise NotImplementedError


class AlwaysHead(RBLHead):
    logical_type = "ALWAYS"

    def __init__(self, d_model: int, d_head: int, init_beta: float = 2.0) -> None:
        super().__init__(d_model, d_head)
        self.kernel = AlwaysKernel(init_beta=init_beta)

    def forward(
        self,
        x: Tensor,
        causal: bool = False,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q, k, v = self._project(x)
        q, k = self._apply_rotary_qk(q, k, cos, sin)
        if q_gain is not None:
            q = q * q_gain.to(dtype=q.dtype)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = self.kernel(scores)
        if causal:
            weights = self._apply_causal(weights)
        output = torch.bmm(weights, v)
        return output, weights

    def explain(self) -> str:
        beta = float(self.kernel.beta.detach())
        return f"G (Historically) head: softmin (β={beta:.3f}); β→0 → argmin"


class EventuallyHead(RBLHead):
    logical_type = "EVENTUALLY"

    def __init__(self, d_model: int, d_head: int, init_alpha: float = 8.0) -> None:
        super().__init__(d_model, d_head)
        self.kernel = EventuallyKernel(init_alpha=init_alpha)

    def forward(
        self,
        x: Tensor,
        causal: bool = False,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q, k, v = self._project(x)
        q, k = self._apply_rotary_qk(q, k, cos, sin)
        if q_gain is not None:
            q = q * q_gain.to(dtype=q.dtype)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = self.kernel(scores)
        if causal:
            weights = self._apply_causal(weights)
        output = torch.bmm(weights, v)
        return output, weights

    def explain(self) -> str:
        alpha = float(self.kernel.alpha.detach())
        return f"EVENTUALLY head: sparse/peaked attention (α={alpha:.3f}); high α → argmax position"


class UntilHead(RBLHead):
    logical_type = "UNTIL"

    def __init__(
        self,
        d_model: int,
        d_head: int,
        seq_len: int,
        init_tau_frac: float = 0.5,
    ) -> None:
        super().__init__(d_model, d_head)
        self.kernel = UntilKernel(seq_len=seq_len, init_tau_frac=init_tau_frac)

    def forward(
        self,
        x: Tensor,
        causal: bool = False,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q, k, v = self._project(x)
        q, k = self._apply_rotary_qk(q, k, cos, sin)
        if q_gain is not None:
            q = q * q_gain.to(dtype=q.dtype)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = self.kernel(scores)
        if causal:
            weights = self._apply_causal(weights)
        output = torch.bmm(weights, v)
        return output, weights

    def explain(self) -> str:
        tau = float(self.kernel.tau.detach())
        kappa = float(self.kernel.kappa.detach())
        return f"S (Since) head: window [τ,t] (τ={tau:.2f}, κ={kappa:.3f})"


class NextHead(RBLHead):
    logical_type = "NEXT"

    def __init__(
        self,
        d_model: int,
        d_head: int,
        seq_len: int,
        init_delta: float = 1.0,
        init_sigma: float = 1.0,
    ) -> None:
        super().__init__(d_model, d_head)
        self.kernel = NextKernel(
            seq_len=seq_len, init_delta=init_delta, init_sigma=init_sigma
        )

    def forward(
        self,
        x: Tensor,
        causal: bool = False,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        q_gain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q, k, v = self._project(x)
        q, k = self._apply_rotary_qk(q, k, cos, sin)
        if q_gain is not None:
            q = q * q_gain.to(dtype=q.dtype)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        weights = self.kernel(scores)
        if causal:
            weights = self._apply_causal(weights)
        output = torch.bmm(weights, v)
        return output, weights

    def explain(self) -> str:
        delta = float(self.kernel.delta.detach())
        sigma = float(self.kernel.sigma.detach())
        return f"Y (Yesterday) head: Gaussian at t−δ (δ={delta:.3f}, σ={sigma:.3f})"


HEAD_REGISTRY: dict[str, type[RBLHead]] = {
    "always": AlwaysHead,
    "eventually": EventuallyHead,
    "until": UntilHead,
    "next": NextHead,
}


def make_head(head_type: str, d_model: int, d_head: int, seq_len: int) -> RBLHead:
    key = head_type.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type '{head_type}'. Available: {list(HEAD_REGISTRY)}")
    cls = HEAD_REGISTRY[key]
    if cls in (UntilHead, NextHead):
        return cls(d_model=d_model, d_head=d_head, seq_len=seq_len)
    return cls(d_model=d_model, d_head=d_head)
