# -*- coding: utf-8 -*-
"""
Key idea:
  - Compute a k-hop node subset around focus_nodes (undirected for coverage).
  - Induce a directed subgraph using the original CFG edge_index (keep edge types).
  - Build an edit_mask for inpainting diffusion: only nodes within edit_hops are editable.

Requires:
  torch, torch_geometric
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e


# ---------
# Utilities
# ---------

def _as_long_tensor(x: Union[torch.Tensor, Sequence[int]], device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.long()
    else:
        t = torch.tensor(list(x), dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t


def _multi_source_bfs_dist(
    edge_index_undirected: torch.Tensor,
    num_nodes: int,
    sources: torch.Tensor,
    max_hops: int,
) -> torch.Tensor:
    """
    Compute min hop distance to any node in sources, capped at max_hops+1.
    Returns:
      dist: Long[num_nodes], dist[i] in {0..max_hops} if reachable, else max_hops+1
    """
    dist = torch.full((num_nodes,), fill_value=max_hops + 1, dtype=torch.long)

    if sources.numel() == 0:
        return dist

    # adjacency list on CPU for simplicity (graphs are small-ish in CFG PoC)
    ei = edge_index_undirected.cpu()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in ei.t().tolist():
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)

    from collections import deque
    q = deque()

    src_list = sources.unique().cpu().tolist()
    for s in src_list:
        if 0 <= s < num_nodes:
            dist[s] = 0
            q.append(s)

    while q:
        u = q.popleft()
        du = int(dist[u].item())
        if du >= max_hops:
            continue
        for v in adj[u]:
            if dist[v] > du + 1:
                dist[v] = du + 1
                q.append(v)

    return dist


def attach_security_signals(
    data: Data,
    *,
    risk_score: float,
    risk_score_log: float,
    safe: bool,
    cond_tags: Sequence[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Data:
    """
    Attach A2 signals to a PyG Data (safe for training: numeric as tensors, tags as list).
    """
    data.risk_score = torch.tensor([float(risk_score)], dtype=torch.float32)
    data.risk_score_log = torch.tensor([float(risk_score_log)], dtype=torch.float32)
    data.safe_label = torch.tensor([1 if safe else 0], dtype=torch.long)
    data.cond_tags = list(cond_tags)

    if extra:
        for k, v in extra.items():
            setattr(data, k, v)
    return data


# -------------------------------
# Focus-subgraph + inpainting mask
# -------------------------------

def extract_focus_subgraph(
    data: Data,
    *,
    focus_nodes: Union[torch.Tensor, Sequence[int]],
    num_hops: int = 2,
    edit_hops: int = 1,
    undirected_for_hops: bool = True,
    empty_focus_policy: str = "all",  # {"all","none"}
    keep_code_str: bool = True,
) -> Data:
    """
    Build a k-hop induced subgraph around focus nodes and create an edit mask.

    Args:
      data: PyG Data from Step A4
      focus_nodes: node indices in the *original* graph (PyG indexing)
      num_hops: size of extracted neighborhood
      edit_hops: editable radius (<= num_hops), nodes within edit_hops become edit_mask=True
      undirected_for_hops: compute k-hop subset on undirected version for better coverage
      empty_focus_policy:
        - "all": if no focus nodes, treat all nodes as focus (useful for safe samples / pretraining)
        - "none": keep empty; returns the full graph with edit_mask all False
      keep_code_str: if original data has code_str, keep it sliced in the subgraph

    Returns:
      sub_data: Data with fields:
        node_type, edge_index, edge_type, lineno, end_lineno, num_nodes
        focus_nodes (subgraph indexing)
        edit_mask (Bool[N_sub])
        dist_to_focus (Long[N_sub])
        plus any existing scalar fields like risk_score / cond_tags if already attached.
    """
    assert num_hops >= 0
    assert edit_hops >= 0
    assert edit_hops <= num_hops, "edit_hops should be <= num_hops"

    device = data.edge_index.device
    N = int(data.num_nodes)

    focus = _as_long_tensor(focus_nodes, device=device)
    focus = focus[(focus >= 0) & (focus < N)].unique()

    if focus.numel() == 0:
        if empty_focus_policy == "all":
            focus = torch.arange(N, device=device, dtype=torch.long)
        elif empty_focus_policy == "none":
            # no focus => no edit; return full graph with all-false edit_mask
            sub_data = data.clone()
            sub_data.focus_nodes = torch.zeros((0,), dtype=torch.long, device=device)
            sub_data.edit_mask = torch.zeros((N,), dtype=torch.bool, device=device)
            sub_data.dist_to_focus = torch.full((N,), fill_value=num_hops + 1, dtype=torch.long, device=device)
            return sub_data
        else:
            raise ValueError(f"Unknown empty_focus_policy: {empty_focus_policy}")

    # 1) Compute node subset (k-hop) using undirected graph for coverage
    hop_edge_index = data.edge_index
    if undirected_for_hops:
        hop_edge_index = to_undirected(hop_edge_index, num_nodes=N)

    # relabel_nodes=False => subset are original node ids; mapping are positions of focus nodes in subset
    subset, _, mapping, _ = k_hop_subgraph(
        focus,
        num_hops,
        hop_edge_index,
        relabel_nodes=False,
        num_nodes=N,
    )
    subset = subset.unique()

    # 2) Induce directed subgraph from original edge_index, relabel nodes by subset ordering
    # return_edge_mask to slice edge_type
    edge_index_sub, edge_mask = subgraph(
        subset,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=N,
        return_edge_mask=True,
    )

    # map focus nodes into subgraph indices:
    # mapping returned by k_hop_subgraph corresponds to positions in `subset` ordering when relabel_nodes=False.
    # However `subset` may have been uniqued; ensure consistent by recomputing focus positions:
    # build an inverse map: original_id -> new_id
    inv = -torch.ones((N,), dtype=torch.long, device=device)
    inv[subset] = torch.arange(subset.numel(), device=device, dtype=torch.long)
    focus_sub = inv[focus]
    focus_sub = focus_sub[focus_sub >= 0].unique()

    # 3) Slice node attributes
    def _slice_attr(attr_name: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if hasattr(data, attr_name):
            t = getattr(data, attr_name)
            if isinstance(t, torch.Tensor) and t.size(0) == N:
                return t[subset]
        return default

    node_type_sub = _slice_attr("node_type")
    lineno_sub = _slice_attr("lineno")
    end_lineno_sub = _slice_attr("end_lineno")

    if node_type_sub is None:
        raise ValueError("data.node_type is required for diffusion; missing in input Data.")

    # 4) Slice edge_type
    edge_type_sub = None
    if hasattr(data, "edge_type"):
        et = getattr(data, "edge_type")
        if isinstance(et, torch.Tensor) and et.numel() == int(data.edge_index.size(1)):
            edge_type_sub = et[edge_mask]
    if edge_type_sub is None:
        edge_type_sub = torch.zeros((edge_index_sub.size(1),), dtype=torch.long, device=device)

    # 5) Compute dist_to_focus (on the induced subgraph, undirected)
    # make undirected edges for BFS
    edge_index_sub_und = to_undirected(edge_index_sub, num_nodes=subset.numel())
    dist = _multi_source_bfs_dist(edge_index_sub_und, int(subset.numel()), focus_sub, max_hops=num_hops).to(device)

    # 6) Build edit_mask: editable nodes are those within edit_hops of focus
    edit_mask = dist <= int(edit_hops)

    # 7) Build sub_data
    sub_data = Data(
        node_type=node_type_sub,
        edge_index=edge_index_sub,
        edge_type=edge_type_sub,
        num_nodes=int(subset.numel()),
    )
    if lineno_sub is not None:
        sub_data.lineno = lineno_sub
    if end_lineno_sub is not None:
        sub_data.end_lineno = end_lineno_sub

    sub_data.focus_nodes = focus_sub
    sub_data.edit_mask = edit_mask
    sub_data.dist_to_focus = dist

    # carry over vocab sizes if present
    if hasattr(data, "node_vocab_size"):
        sub_data.node_vocab_size = getattr(data, "node_vocab_size")
    if hasattr(data, "edge_vocab_size"):
        sub_data.edge_vocab_size = getattr(data, "edge_vocab_size")

    # carry over security signals if already attached (risk_score, cond_tags, etc.)
    for k in ["risk_score", "risk_score_log", "safe_label", "cond_tags"]:
        if hasattr(data, k):
            setattr(sub_data, k, getattr(data, k))

    # optionally carry code_str for debugging
    if keep_code_str and hasattr(data, "code_str"):
        cs = getattr(data, "code_str")
        if isinstance(cs, list) and len(cs) == N:
            sub_data.code_str = [cs[i] for i in subset.tolist()]

    # keep mapping for later (optional but useful):
    # original node id for each subgraph node position
    sub_data.orig_node_ids = subset.clone()

    return sub_data


# -------------------------------
# Minimal end-to-end glue (optional)
# -------------------------------

def focus_lines_to_focus_nodes_pyg(
    focus_lines: Sequence[int],
    line_to_nodes_nx: Dict[int, List[int]],
    nx_to_pyg_node_map: Dict[int, int],
) -> torch.Tensor:
    """
    Convert A2 focus_lines -> (nx node ids via line_to_nodes) -> pyg node indices.
    """
    focus_nx = set()
    for ln in focus_lines:
        focus_nx.update(line_to_nodes_nx.get(int(ln), []))
    focus_pyg = [nx_to_pyg_node_map[n] for n in focus_nx if n in nx_to_pyg_node_map]
    if not focus_pyg:
        return torch.zeros((0,), dtype=torch.long)
    return torch.tensor(sorted(set(focus_pyg)), dtype=torch.long)


if __name__ == "__main__":
    # quick sanity check with Step A3/A4
    from .ast_cfg_python import build_cfg_from_code
    from .pyg_convert import nx_cfg_to_pyg

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

    # fake focus_lines
    focus_lines = [4, 8]
    focus_nodes = focus_lines_to_focus_nodes_pyg(focus_lines, line_to_nodes, node_map)

    sub = extract_focus_subgraph(
        data,
        focus_nodes=focus_nodes,
        num_hops=2,
        edit_hops=1,
        empty_focus_policy="all",
    )
    print(sub)
    print("focus_nodes(sub):", sub.focus_nodes.tolist())
    print("editable nodes:", int(sub.edit_mask.sum().item()), "/", sub.num_nodes)

