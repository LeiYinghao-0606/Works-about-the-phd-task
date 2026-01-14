# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Standard sinusoidal embedding used in diffusion.
    t: Long/Float tensor of shape [B]
    return: Float tensor [B, dim]
    """
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None] * freqs[None, :]  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.size(0), 1), device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """
    t_graph [B] -> time vector [B, hidden_dim]
    """
    def __init__(self, hidden_dim: int, time_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        time_dim = int(time_dim or hidden_dim)
        self.time_dim = time_dim
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_graph: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(t_graph, self.time_dim)
        return self.mlp(emb)  # [B, hidden_dim]


class MaskEmbedding(nn.Module):
    """
    Optional: embed edit_mask (0/1) into a vector, to tell the denoiser which nodes are editable.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(2, hidden_dim)

    def forward(self, mask01: torch.Tensor) -> torch.Tensor:
        return self.emb(mask01.long())

