# -*- coding: utf-8 -*-
"""
Requires:
  pip install torch torch-geometric

Outputs:
  data.node_type: Long[N]
  data.edge_index: Long[2, E]
  data.edge_type: Long[E]
  data.lineno: Long[N]
  data.end_lineno: Long[N]
  data.num_nodes: N

Optionally keep:
  data.code_str: List[str]  (NOT tensor) for debugging

Also returns:
  node_id_map: Dict[nx_node_id -> pyg_index]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError(
        "torch_geometric is required. Install via: pip install torch-geometric"
    ) from e


# ---------------------------
# Vocabularies
# ---------------------------

DEFAULT_NODE_VOCAB = [
    # special
    "UNK",
    "ENTRY",
    "EXIT",
    # control
    "If",
    "For",
    "While",
    "Try",
    # terminals
    "Return",
    "Raise",
    "Break",
    "Continue",
    # defs
    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    # generic stmt fallbacks
    "Assign",
    "AnnAssign",
    "AugAssign",
    "Expr",
    "Call",        # often embedded in Expr; kept for future extensions
    "Import",
    "ImportFrom",
    "With",
    "AsyncWith",
    "Global",
    "Nonlocal",
    "Pass",
    "Assert",
    "Delete",
]

DEFAULT_EDGE_VOCAB = [
    "NEXT",
    "T",
    "F",
    "BACK",
    "BREAK",
    "CONTINUE",
    "UNK",
]


@dataclass
class Vocab:
    node2id: Dict[str, int]
    edge2id: Dict[str, int]

    @classmethod
    def from_lists(cls, node_vocab: List[str], edge_vocab: List[str]) -> "Vocab":
        node2id = {t: i for i, t in enumerate(node_vocab)}
        edge2id = {t: i for i, t in enumerate(edge_vocab)}
        return cls(node2id=node2id, edge2id=edge2id)

    def node_id(self, t: Optional[str]) -> int:
        if not t:
            return self.node2id.get("UNK", 0)
        return self.node2id.get(t, self.node2id.get("UNK", 0))

    def edge_id(self, t: Optional[str]) -> int:
        if not t:
            return self.edge2id.get("UNK", len(self.edge2id) - 1)
        return self.edge2id.get(t, self.edge2id.get("UNK", len(self.edge2id) - 1))


# ---------------------------
# Conversion
# ---------------------------

def nx_cfg_to_pyg(
    g: nx.DiGraph,
    *,
    vocab: Optional[Vocab] = None,
    keep_code_str: bool = True,
    sort_nodes: bool = True,
) -> Tuple[Data, Dict[int, int]]:
    """
    Convert a NetworkX DiGraph CFG to PyG Data.

    Args:
      g: nx.DiGraph produced by Step A3
      vocab: node/edge vocab mapping
      keep_code_str: store code snippets in Data (non-tensor)
      sort_nodes: deterministic ordering (by node id)

    Returns:
      data: torch_geometric.data.Data
      node_id_map: dict {nx_node_id: pyg_index}
    """
    if vocab is None:
        vocab = Vocab.from_lists(DEFAULT_NODE_VOCAB, DEFAULT_EDGE_VOCAB)

    # Node ordering
    nx_nodes = list(g.nodes())
    if sort_nodes:
        try:
            nx_nodes = sorted(nx_nodes)
        except Exception:
            pass

    node_id_map: Dict[int, int] = {nid: i for i, nid in enumerate(nx_nodes)}
    N = len(nx_nodes)

    # Node tensors
    node_type = torch.zeros((N,), dtype=torch.long)
    lineno = torch.zeros((N,), dtype=torch.long)
    end_lineno = torch.zeros((N,), dtype=torch.long)

    code_str: List[str] = [""] * N

    for nid, i in node_id_map.items():
        attrs = g.nodes[nid]
        ast_type = attrs.get("ast_type", None)
        node_type[i] = vocab.node_id(ast_type)

        ln = attrs.get("lineno", None)
        eln = attrs.get("end_lineno", None)
        lineno[i] = int(ln) if isinstance(ln, int) and ln > 0 else 0
        end_lineno[i] = int(eln) if isinstance(eln, int) and eln > 0 else 0

        if keep_code_str:
            c = attrs.get("code", "") or ""
            code_str[i] = str(c)

    # Edge tensors
    edges_u: List[int] = []
    edges_v: List[int] = []
    edge_type_list: List[int] = []

    for u, v, attrs in g.edges(data=True):
        if u not in node_id_map or v not in node_id_map:
            continue
        et = attrs.get("etype", None)
        edges_u.append(node_id_map[u])
        edges_v.append(node_id_map[v])
        edge_type_list.append(vocab.edge_id(et))

    if len(edges_u) == 0:
        # avoid empty edge_index issues
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    data = Data(
        node_type=node_type,
        edge_index=edge_index,
        edge_type=edge_type,
        lineno=lineno,
        end_lineno=end_lineno,
        num_nodes=N,
    )

    if keep_code_str:
        # non-tensor attributes are allowed in PyG Data; keep for debug only
        data.code_str = code_str

    # Store vocab size info (useful for diffusion configs)
    data.node_vocab_size = len(vocab.node2id)
    data.edge_vocab_size = len(vocab.edge2id)

    return data, node_id_map


def focus_nodes_to_index(
    focus_nodes_nx: List[int],
    node_id_map: Dict[int, int],
) -> torch.Tensor:
    """
    Convert nx focus node ids -> pyg indices tensor.
    """
    idx = [node_id_map[n] for n in focus_nodes_nx if n in node_id_map]
    if not idx:
        return torch.zeros((0,), dtype=torch.long)
    return torch.tensor(sorted(set(idx)), dtype=torch.long)


# ---------------------------
# Quick test
# ---------------------------
if __name__ == "__main__":
    from .ast_cfg_python import build_cfg_from_code

    code = """
def f(x):
    if x > 0:
        print("pos")
    else:
        print("neg")
    for i in range(3):
        if i == 1:
            continue
        if i == 2:
            break
        print(i)
    return 1
"""
    g, line_to_nodes = build_cfg_from_code(code)
    data, node_map = nx_cfg_to_pyg(g)
    print(data)
    print("node_type shape:", data.node_type.shape, "edge_index shape:", data.edge_index.shape)

