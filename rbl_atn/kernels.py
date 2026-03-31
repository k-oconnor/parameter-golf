"""Attention shape kernels — one per past-LTL operator (paper-aligned)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HistoricallyKernel(nn.Module):
    def __init__(self, init_beta: float = 2.0):
        super().__init__()
        init_log = math.log(math.exp(max(init_beta - 1.0, 1e-3)) - 1.0)
        self.log_beta = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))

    @property
    def beta(self) -> Tensor:
        return F.softplus(self.log_beta) + 1.0

    def forward(self, scores: Tensor) -> Tensor:
        return F.softmax(-scores / self.beta, dim=-1)

    def beta_regularizer(self, lambda_: float = 0.01) -> Tensor:
        return lambda_ / self.beta


class EventuallyKernel(nn.Module):
    def __init__(self, init_alpha: float = 8.0):
        super().__init__()
        init_log = math.log(math.exp(max(init_alpha - 1.0, 1e-3)) - 1.0)
        self.log_alpha = nn.Parameter(torch.tensor(init_log, dtype=torch.float32))

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.log_alpha) + 1.0

    def forward(self, scores: Tensor) -> Tensor:
        return F.softmax(self.alpha * scores, dim=-1)

    def sparsity_penalty(self) -> Tensor:
        return F.relu(5.0 - self.alpha)


class SinceKernel(nn.Module):
    def __init__(self, seq_len: int, init_tau_frac: float = 0.5):
        super().__init__()
        self.seq_len = seq_len
        init_tau = math.log(init_tau_frac / (1.0 - init_tau_frac + 1e-8))
        self.tau_param = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(1.0), dtype=torch.float32))

    @property
    def tau(self) -> Tensor:
        return torch.sigmoid(self.tau_param) * self.seq_len

    @property
    def kappa(self) -> Tensor:
        return F.softplus(self.log_kappa) + 0.1

    def forward(self, scores: Tensor) -> Tensor:
        T = scores.shape[-1]
        device, dtype = scores.device, scores.dtype
        u = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
        tau = self.tau
        k = self.kappa
        causal = (u <= t).to(dtype)
        rise = torch.sigmoid(k * (u - tau))
        fall = torch.sigmoid(k * (t - u))
        kernel = rise * fall * causal
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        log_kernel = torch.log(kernel.clamp(min=1e-8)).unsqueeze(0)
        return F.softmax(scores + log_kernel, dim=-1)


class YesterdayKernel(nn.Module):
    def __init__(self, seq_len: int, init_delta: float = 1.0, init_sigma: float = 1.0):
        super().__init__()
        self.seq_len = seq_len
        self.delta = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))
        init_log_sigma = math.log(math.exp(max(init_sigma - 0.5, 1e-3)) - 1.0)
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=torch.float32))

    @property
    def sigma(self) -> Tensor:
        return F.softplus(self.log_sigma) + 0.5

    def _kernel_weights(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        q_pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
        k_pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)
        diff = k_pos - q_pos + self.delta
        return torch.exp(-(diff**2) / (2.0 * self.sigma**2))

    def forward(self, scores: Tensor) -> Tensor:
        T = scores.shape[-1]
        kernel_w = self._kernel_weights(T, scores.device, scores.dtype)
        log_kernel = torch.log(kernel_w.clamp(min=1e-8)).unsqueeze(0)
        return F.softmax(scores + log_kernel, dim=-1)


AlwaysKernel = HistoricallyKernel
UntilKernel = SinceKernel
NextKernel = YesterdayKernel
