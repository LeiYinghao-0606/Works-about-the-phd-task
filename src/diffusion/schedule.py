# -*- coding: utf-8 -*-
"""
Discrete diffusion noise schedule (Step C1)

We use a simple "uniform replacement" corruption:
  At each step t:
    with prob beta_t: replace token with a random category (uniform)
    with prob 1-beta_t: keep previous token
This yields a closed-form:
  P(x_t = x_0) = alpha_bar_t = prod_{s=1..t} (1 - beta_s)
  Otherwise token is approximately uniform.

This schedule is stable, easy to implement, and works well for inpainting-style discrete diffusion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NoiseSchedule:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: Optional[str] = None

    def __post_init__(self):
        assert self.T >= 1
        assert 0.0 < self.beta_start < 1.0
        assert 0.0 < self.beta_end < 1.0

        betas = torch.linspace(self.beta_start, self.beta_end, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        if self.device is not None:
            self.to(self.device)

    def to(self, device: str) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.device = device
        return self

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: Long tensor with values in [1..T]
        returns alpha_bar_t in [0,1], same shape as t (float)
        """
        assert t.dtype in (torch.int32, torch.int64)
        # convert to 0-index
        idx = torch.clamp(t - 1, 0, self.T - 1)
        return self.alpha_bars[idx]

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        beta_t for t in [1..T]
        """
        idx = torch.clamp(t - 1, 0, self.T - 1)
        return self.betas[idx]

