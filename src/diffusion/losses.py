# -*- coding: utf-8 -*-
"""
Diffusion training losses (masked categorical CE etc.).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,        # [N, V]
    target: torch.Tensor,        # [N]
    mask: torch.Tensor,          # [N] bool
) -> torch.Tensor:
    """
    Mean CE over masked positions. Returns 0 if mask is empty.
    """
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    loss_all = F.cross_entropy(logits, target, reduction="none")
    return (loss_all * mask.float()).sum() / (mask.float().sum() + 1e-12)


def kl_div_categorical(
    p_logits: torch.Tensor,      # [N, V]
    q_logits: torch.Tensor,      # [N, V]
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KL(softmax(p) || softmax(q)) averaged (optionally masked).
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    kl = F.kl_div(p, q, reduction="none").sum(dim=-1)  # [N]
    if mask is None:
        return kl.mean()
    if mask.sum().item() == 0:
        return torch.zeros((), device=p_logits.device, dtype=torch.float32)
    return (kl * mask.float()).sum() / (mask.float().sum() + 1e-12)
