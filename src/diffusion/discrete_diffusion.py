# -*- coding: utf-8 -*-
"""
Training objective:
  - Sample timestep t ~ Uniform{1..T} per graph.
  - Corrupt x0 -> xt with closed-form uniform replacement using alpha_bar(t).
  - Denoiser predicts logits for x0 (node and edge categories).
  - Loss = masked cross-entropy on editable region (edit_mask) only.

Inpainting:
  - Non-editable tokens are always clamped to original x0 in both q_sample and sampling loop.

We assume edge_index is fixed; we only diffuse node_type and edge_type.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from .schedule import NoiseSchedule


def _ensure_batch_vector(data: Data) -> torch.Tensor:
    """
    Ensure we have data.batch (node -> graph id).
    If missing, treat as a single-graph batch.
    """
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros((data.num_nodes,), dtype=torch.long, device=data.node_type.device)


def _edge_batch(edge_index: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    """
    edge belongs to the graph of its source node (works for disjoint batched graphs).
    """
    return node_batch[edge_index[0]]


def _default_edge_edit_mask(data: Data, node_edit_mask: torch.Tensor) -> torch.Tensor:
    """
    Edge is editable if either endpoint is editable (inpainting region touches the edge).
    """
    ei = data.edge_index
    return node_edit_mask[ei[0]] | node_edit_mask[ei[1]]


def _sample_uniform(shape, vocab_size: int, device) -> torch.Tensor:
    return torch.randint(low=0, high=vocab_size, size=shape, device=device, dtype=torch.long)


def _corrupt_categorical_x0_to_xt(
    x0: torch.Tensor,
    alpha_bar_elem: torch.Tensor,
    vocab_size: int,
    editable_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form corruption:
      with prob alpha_bar: keep x0
      else: uniform random token
    Only applied where editable_mask=True; other positions remain x0.
    """
    assert x0.dtype == torch.long
    device = x0.device
    keep = torch.rand_like(alpha_bar_elem) < alpha_bar_elem  # bool

    rand_tok = _sample_uniform(x0.shape, vocab_size, device=device)
    xt = torch.where(keep, x0, rand_tok)

    # inpainting clamp
    xt = torch.where(editable_mask, xt, x0)
    return xt


def _masked_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [N, V], target: [N], mask: [N] bool
    return mean CE over masked positions (safe when mask sum==0).
    """
    if mask.sum().item() == 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    loss_all = F.cross_entropy(logits, target, reduction="none")  # [N]
    loss = (loss_all * mask.float()).sum() / (mask.float().sum() + 1e-12)
    return loss


@dataclass
class DiffusionBatch:
    """
    Container for training step outputs.
    """
    t_graph: torch.Tensor              # [B] timesteps in 1..T
    node_xt: torch.Tensor              # [N]
    edge_xt: torch.Tensor              # [E]
    node_edit_mask: torch.Tensor       # [N] bool
    edge_edit_mask: torch.Tensor       # [E] bool


class DiscreteInpaintDiffusion:
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        schedule: NoiseSchedule,
    ):
        assert node_vocab_size > 1 and edge_vocab_size > 1
        self.node_vocab_size = int(node_vocab_size)
        self.edge_vocab_size = int(edge_vocab_size)
        self.sched = schedule

    # -----------------------
    # Training-time utilities
    # -----------------------

    def sample_timesteps(self, num_graphs: int, device) -> torch.Tensor:
        """
        Sample t ~ Uniform{1..T} per graph.
        """
        return torch.randint(1, self.sched.T + 1, (num_graphs,), device=device, dtype=torch.long)

    def q_sample(self, data: Data, t_graph: torch.Tensor) -> DiffusionBatch:
        """
        Corrupt x0 -> xt for nodes and edges with closed-form alpha_bar(t).
        Requires:
          data.node_type: [N] long
          data.edge_type: [E] long
          data.edit_mask: [N] bool  (from Step A5)
        """
        assert hasattr(data, "node_type") and hasattr(data, "edge_type")
        assert hasattr(data, "edit_mask"), "Data must contain edit_mask (Step A5)."

        node_batch = _ensure_batch_vector(data)  # [N]
        B = int(node_batch.max().item() + 1)

        assert t_graph.shape == (B,), f"t_graph should be shape [B={B}]"

        # elementwise alpha_bar for each node / edge
        alpha_bar_g = self.sched.alpha_bar(t_graph)  # [B] float
        alpha_bar_node = alpha_bar_g[node_batch]     # [N]

        node_edit_mask = data.edit_mask.bool()       # [N]
        node_xt = _corrupt_categorical_x0_to_xt(
            x0=data.node_type.long(),
            alpha_bar_elem=alpha_bar_node,
            vocab_size=self.node_vocab_size,
            editable_mask=node_edit_mask,
        )

        # edges
        e_batch = _edge_batch(data.edge_index, node_batch)  # [E]
        alpha_bar_edge = alpha_bar_g[e_batch]               # [E]

        if hasattr(data, "edge_edit_mask"):
            edge_edit_mask = data.edge_edit_mask.bool()
        else:
            edge_edit_mask = _default_edge_edit_mask(data, node_edit_mask)

        edge_xt = _corrupt_categorical_x0_to_xt(
            x0=data.edge_type.long(),
            alpha_bar_elem=alpha_bar_edge,
            vocab_size=self.edge_vocab_size,
            editable_mask=edge_edit_mask,
        )

        return DiffusionBatch(
            t_graph=t_graph,
            node_xt=node_xt,
            edge_xt=edge_xt,
            node_edit_mask=node_edit_mask,
            edge_edit_mask=edge_edit_mask,
        )

    def make_noisy_data(self, data: Data, diff: DiffusionBatch) -> Data:
        """
        Create a shallow copy of data with noisy node/edge tokens.
        """
        noisy = data.clone()
        noisy.node_type = diff.node_xt
        noisy.edge_type = diff.edge_xt
        noisy.t_graph = diff.t_graph  # store for convenience
        return noisy

    def training_losses(
        self,
        model,
        data: Data,
        *,
        lambda_node: float = 1.0,
        lambda_edge: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        One training forward:
          - sample t
          - q_sample -> noisy data
          - model predicts logits for x0
          - masked CE loss on editable region

        Model contract (for now):
          node_logits, edge_logits = model(noisy_data, t_graph)
        where:
          node_logits: [N, node_vocab_size]
          edge_logits: [E, edge_vocab_size]
        """
        device = data.node_type.device
        node_batch = _ensure_batch_vector(data)
        B = int(node_batch.max().item() + 1)
        t_graph = self.sample_timesteps(B, device=device)

        diff = self.q_sample(data, t_graph)
        noisy = self.make_noisy_data(data, diff)

        node_logits, edge_logits = model(noisy, t_graph)

        # targets are original x0
        node_loss = _masked_ce_loss(node_logits, data.node_type.long(), diff.node_edit_mask)
        edge_loss = _masked_ce_loss(edge_logits, data.edge_type.long(), diff.edge_edit_mask)
        total = lambda_node * node_loss + lambda_edge * edge_loss

        return {
            "loss": total,
            "node_loss": node_loss.detach(),
            "edge_loss": edge_loss.detach(),
            "t_mean": t_graph.float().mean().detach(),
        }

    # -----------------------
    # Sampling (inpainting) - minimal heuristic loop
    # -----------------------
    @torch.no_grad()
    def sample_inpaint(
        self,
        model,
        data: Data,
        *,
        steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Data:
        """
        Minimal reverse loop for inpainting (heuristic but practical for PoC):

        Start at t=T:
          editable positions set to random tokens, fixed positions clamped to x0.
        For t = T..1:
          model predicts x0 logits -> sample x0_hat
          set x_{t-1} = x0_hat with prob (1 - beta_t), else random (on editable only)
          clamp fixed positions back to x0

        This is sufficient to get "improved graphs" candidates; later we can refine to a more principled sampler.
        """
        assert hasattr(data, "edit_mask")
        device = data.node_type.device
        node_batch = _ensure_batch_vector(data)
        B = int(node_batch.max().item() + 1)

        T = self.sched.T if steps is None else int(steps)
        T = min(T, self.sched.T)

        # init xt
        node_edit = data.edit_mask.bool()
        if hasattr(data, "edge_edit_mask"):
            edge_edit = data.edge_edit_mask.bool()
        else:
            edge_edit = _default_edge_edit_mask(data, node_edit)

        node_xt = data.node_type.clone()
        edge_xt = data.edge_type.clone()

        node_xt = torch.where(node_edit, _sample_uniform(node_xt.shape, self.node_vocab_size, device), node_xt)
        edge_xt = torch.where(edge_edit, _sample_uniform(edge_xt.shape, self.edge_vocab_size, device), edge_xt)

        cur = data.clone()
        cur.node_type = node_xt
        cur.edge_type = edge_xt

        for t in range(T, 0, -1):
            t_graph = torch.full((B,), fill_value=t, device=device, dtype=torch.long)
            node_logits, edge_logits = model(cur, t_graph)

            # sample x0_hat
            node_probs = F.softmax(node_logits / max(1e-8, temperature), dim=-1)
            edge_probs = F.softmax(edge_logits / max(1e-8, temperature), dim=-1)
            node_x0_hat = torch.multinomial(node_probs, num_samples=1).squeeze(-1)
            edge_x0_hat = torch.multinomial(edge_probs, num_samples=1).squeeze(-1)

            beta_t = self.sched.beta(t_graph)  # [B]
            beta_node = beta_t[node_batch]     # [N]
            e_batch = _edge_batch(cur.edge_index, node_batch)
            beta_edge = beta_t[e_batch]        # [E]

            # update rule: with prob (1-beta) take x0_hat else random (only editable)
            take_hat_node = (torch.rand_like(beta_node) >= beta_node)
            take_hat_edge = (torch.rand_like(beta_edge) >= beta_edge)

            node_next = torch.where(take_hat_node, node_x0_hat, _sample_uniform(node_x0_hat.shape, self.node_vocab_size, device))
            edge_next = torch.where(take_hat_edge, edge_x0_hat, _sample_uniform(edge_x0_hat.shape, self.edge_vocab_size, device))

            # clamp
            node_next = torch.where(node_edit, node_next, data.node_type)
            edge_next = torch.where(edge_edit, edge_next, data.edge_type)

            cur.node_type = node_next
            cur.edge_type = edge_next

        return cur

