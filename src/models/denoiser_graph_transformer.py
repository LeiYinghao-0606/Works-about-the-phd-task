# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from .embeddings import TimeEmbedding, MaskEmbedding


def _ensure_batch(data) -> torch.Tensor:
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros((data.num_nodes,), dtype=torch.long, device=data.node_type.device)


def _edge_batch(edge_index: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
    return node_batch[edge_index[0]]


@dataclass
class DenoiserConfig:
    node_vocab_size: int
    edge_vocab_size: int

    hidden_dim: int = 192
    num_layers: int = 4
    heads: int = 4
    dropout: float = 0.1

    time_dim: Optional[int] = None  # if None, use hidden_dim
    use_edit_mask_embedding: bool = True


class GraphTransformerDenoiser(nn.Module):
    """
    Denoiser for discrete diffusion on graphs with fixed topology.

    Input (noisy Data):
      - node_type: [N] long
      - edge_type: [E] long
      - edge_index: [2, E]
      - edit_mask: [N] bool (optional but recommended)

    Forward signature matches Step C1:
      node_logits, edge_logits = model(noisy_data, t_graph)

    Outputs:
      - node_logits: [N, node_vocab_size]
      - edge_logits: [E, edge_vocab_size]
    """

    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg

        H = cfg.hidden_dim
        assert cfg.node_vocab_size > 1 and cfg.edge_vocab_size > 1
        assert cfg.num_layers >= 1
        assert cfg.heads >= 1
        assert H % cfg.heads == 0, "hidden_dim must be divisible by heads"

        # Token embeddings
        self.node_emb = nn.Embedding(cfg.node_vocab_size, H)
        self.edge_emb = nn.Embedding(cfg.edge_vocab_size, H)

        # Time embedding (per-graph)
        self.time_emb = TimeEmbedding(hidden_dim=H, time_dim=cfg.time_dim, dropout=cfg.dropout)

        # Optional: tell network which nodes are editable
        self.use_mask = bool(cfg.use_edit_mask_embedding)
        self.mask_emb = MaskEmbedding(H) if self.use_mask else None

        # Graph Transformer layers (edge_attr supported)
        # TransformerConv uses attention; edge_dim must match edge_attr dim
        self.convs = nn.ModuleList([
            TransformerConv(
                in_channels=H,
                out_channels=H // cfg.heads,
                heads=cfg.heads,
                dropout=cfg.dropout,
                edge_dim=H,
                beta=True,  # enables residual gating inside TransformerConv (if supported)
            )
            for _ in range(cfg.num_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(cfg.num_layers)])

        # Output heads
        self.node_out = nn.Linear(H, cfg.node_vocab_size)

        # Edge logits from endpoints + edge embedding + time
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * H + H + H, H),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H, cfg.edge_vocab_size),
        )

    def forward(self, data, t_graph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        data: PyG Data (possibly batched)
        t_graph: [B] long timesteps (1..T), one per graph in the batch
        """
        node_batch = _ensure_batch(data)  # [N]
        B = int(node_batch.max().item() + 1)
        assert t_graph.shape == (B,), f"t_graph must be [B={B}] but got {tuple(t_graph.shape)}"

        # Time embeddings broadcast to nodes/edges
        t_vec = self.time_emb(t_graph)          # [B, H]
        t_node = t_vec[node_batch]              # [N, H]
        e_batch = _edge_batch(data.edge_index, node_batch)  # [E]
        t_edge = t_vec[e_batch]                 # [E, H]

        # Input embeddings
        x = self.node_emb(data.node_type.long()) + t_node

        if self.use_mask and hasattr(data, "edit_mask"):
            mask01 = data.edit_mask.long().view(-1)
            x = x + self.mask_emb(mask01)

        edge_attr = self.edge_emb(data.edge_type.long()) + t_edge  # [E, H]

        # TransformerConv stack with residual + LayerNorm
        for conv, ln in zip(self.convs, self.norms):
            h = conv(x, data.edge_index, edge_attr=edge_attr)
            h = F.dropout(h, p=self.cfg.dropout, training=self.training)
            x = ln(x + h)

        # Node logits
        node_logits = self.node_out(x)  # [N, Vn]

        # Edge logits (type prediction)
        u = data.edge_index[0]
        v = data.edge_index[1]
        xu = x[u]
        xv = x[v]
        edge_feat = torch.cat([xu, xv, edge_attr, t_edge], dim=-1)  # [E, 4H]
        edge_logits = self.edge_mlp(edge_feat)  # [E, Ve]

        return node_logits, edge_logits

