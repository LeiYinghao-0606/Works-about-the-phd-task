# -*- coding: utf-8 -*-
"""
Dataset schema: define required fields in PyG Data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e


# ---- Required fields for diffusion ----
REQUIRED_FIELDS = [
    "node_type",     # [N] long
    "edge_index",    # [2, E] long
    "edge_type",     # [E] long
    "edit_mask",     # [N] bool
    "lineno",        # [N] long  (node -> source line, 1-index, 0 if unknown)
    "node_vocab_size",
    "edge_vocab_size",
]

# ---- Optional but recommended for the PromSec loop ----
OPTIONAL_FIELDS = [
    "focus_nodes",   # [K] long (node indices)
    "cond_tags",     # List[str] (e.g., Bandit test IDs)
    "file_path",     # str
    "source_code",   # str (if you don't want reading from disk during loop)
    "risk_score_log",
    "safe_label",
]


def validate_data(d: Data) -> None:
    """
    Minimal validation for training.
    """
    for k in REQUIRED_FIELDS:
        if not hasattr(d, k):
            raise ValueError(f"Data missing required field: `{k}`")

    assert d.node_type.dtype in (torch.int64, torch.int32), "node_type must be integer tensor"
    assert d.edge_type.dtype in (torch.int64, torch.int32), "edge_type must be integer tensor"
    assert d.edge_index.dtype in (torch.int64, torch.int32), "edge_index must be integer tensor"
    assert d.edit_mask.dtype == torch.bool, "edit_mask must be bool tensor"
    assert d.lineno.dtype in (torch.int64, torch.int32), "lineno must be integer tensor"

    # shape sanity
    assert d.edge_index.dim() == 2 and d.edge_index.size(0) == 2, "edge_index must be [2, E]"
    assert d.node_type.dim() == 1, "node_type must be [N]"
    assert d.edit_mask.dim() == 1 and d.edit_mask.size(0) == d.node_type.size(0), "edit_mask must be [N]"
    assert d.lineno.dim() == 1 and d.lineno.size(0) == d.node_type.size(0), "lineno must be [N]"


def attach_vocab_sizes(d: Data, node_vocab_size: int, edge_vocab_size: int) -> Data:
    d.node_vocab_size = int(node_vocab_size)
    d.edge_vocab_size = int(edge_vocab_size)
    return d

