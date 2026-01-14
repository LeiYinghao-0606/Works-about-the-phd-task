# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv, global_mean_pool
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e


class GuidanceClassifier(nn.Module):
    """
    Graph-level predictor for:
      - risk_score_log (regression): higher = riskier
      - safe_label (binary): 1 safe, 0 unsafe

    Uses RGCNConv to incorporate discrete edge_type as relation type.
    """

    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert node_vocab_size > 0 and edge_vocab_size > 0
        assert num_layers >= 1

        self.node_emb = nn.Embedding(node_vocab_size, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=edge_vocab_size))

        self.dropout = float(dropout)

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.safe_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 1),  # logits
        )

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          risk_pred: [B] float
          safe_logit: [B] float
        """
        node_type = data.node_type  # [N]
        edge_index = data.edge_index  # [2, E]
        edge_type = data.edge_type  # [E]

        x = self.node_emb(node_type)

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        g = global_mean_pool(x, batch)  # [B, H]

        risk_pred = self.risk_head(g).squeeze(-1)     # [B]
        safe_logit = self.safe_head(g).squeeze(-1)    # [B]
        return risk_pred, safe_logit

