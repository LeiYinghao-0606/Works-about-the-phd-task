# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Optional
import random
import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e


def load_pyg_list(path: str) -> List["Data"]:
    """
    Expect torch.save(list_of_Data, path)
    Also supports dict with keys like {'data': list} or {'data_list': list}.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data", "data_list", "graphs", "items"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(f"Unsupported dataset format at {path}. Got type={type(obj)}")


def train_val_split(
    data_list: List["Data"],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List["Data"], List["Data"]]:
    assert 0.0 < val_ratio < 1.0
    idx = list(range(len(data_list)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [data_list[i] for i in idx if i not in val_idx]
    val = [data_list[i] for i in idx if i in val_idx]
    return train, val

