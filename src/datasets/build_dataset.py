# -*- coding: utf-8 -*-
"""
build_dataset.py
Raw -> processed dataset builder (PoC baseline).

Input: a directory of Python files (*.py)
Output: torch files containing List[PyG Data] with required schema.

Pipeline:
  1) Parse code -> AST graph (node_type, edge_index, edge_type, lineno)
  2) Run Bandit -> issues -> cond_tags + edit_mask + labels
  3) Build node vocab across dataset
  4) Save processed graphs: graphs_train.pt / graphs_test.pt (or single file)

Usage:
  python -m src.datasets.build_dataset --raw_dir data/raw --out_dir data/processed
"""

from __future__ import annotations

import argparse
import ast
import os
from typing import Dict, List, Tuple, Any

import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.common.io import read_text, torch_save
from src.common.seed import seed_everything
from src.analysis.bandit_runner import run_bandit_on_code
from src.analysis.score import bandit_score
from src.datasets.schema import validate_data, attach_vocab_sizes


def collect_py_files(raw_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            if fn.endswith(".py"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def build_graph_from_code_ast(code: str) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    """
    Build a simple AST graph:
      - node labels: ast node class name
      - edges: parent -> child
      - lineno: node.lineno if exists else 0
    Returns:
      node_labels: List[str] length N
      edges: List[(u,v)] length E
      linenos: List[int] length N
    """
    tree = ast.parse(code)
    node_labels: List[str] = []
    edges: List[Tuple[int, int]] = []
    linenos: List[int] = []

    node_id = 0
    id_map: Dict[int, int] = {}

    def visit(n: ast.AST, parent: int | None = None):
        nonlocal node_id
        cur = node_id
        id_map[id(n)] = cur
        node_id += 1

        node_labels.append(type(n).__name__)
        ln = int(getattr(n, "lineno", 0) or 0)
        linenos.append(ln)

        if parent is not None:
            edges.append((parent, cur))

        for ch in ast.iter_child_nodes(n):
            visit(ch, cur)

    visit(tree, None)
    return node_labels, edges, linenos


def make_edit_mask_from_issue_lines(linenos: List[int], issue_lines: List[int]) -> torch.Tensor:
    issue_set = set(int(x) for x in issue_lines if int(x) > 0)
    m = [(ln in issue_set) for ln in linenos]
    return torch.tensor(m, dtype=torch.bool)


def build_processed_graphs(
    file_paths: List[str],
    *,
    max_files: int = -1,
) -> Tuple[List[Data], Dict[str, int]]:
    """
    Build Data list without vocab ids first; also return node label freq map.
    """
    data_list: List[Data] = []
    freq: Dict[str, int] = {}

    if max_files > 0:
        file_paths = file_paths[:max_files]

    for fp in file_paths:
        code = read_text(fp)
        node_labels, edges, linenos = build_graph_from_code_ast(code)

        # bandit labeling
        br = run_bandit_on_code(code)
        vs = bandit_score(br)
        issue_lines = [it.line_number for it in br.issues]

        edit_mask = make_edit_mask_from_issue_lines(linenos, issue_lines)
        focus_nodes = torch.nonzero(edit_mask, as_tuple=False).view(-1).long()

        # update vocab freq
        for lb in node_labels:
            freq[lb] = freq.get(lb, 0) + 1

        # temporarily store labels as strings; will map to ids later
        d = Data()
        d.node_label_str = node_labels  # temporary python list
        d.lineno = torch.tensor(linenos, dtype=torch.long)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        d.edge_index = edge_index

        # edge_type in this PoC: all edges are one relation type 0
        d.edge_type = torch.zeros((edge_index.size(1),), dtype=torch.long)

        d.edit_mask = edit_mask
        d.focus_nodes = focus_nodes

        # PromSec loop metadata
        d.file_path = fp
        # You may store full source_code (optional; increases size)
        # d.source_code = code

        d.safe_label = torch.tensor([vs.safe_label], dtype=torch.long)
        d.risk_score_log = torch.tensor([vs.risk_score_log], dtype=torch.float32)
        d.cond_tags = vs.cond_tags  # List[str]

        data_list.append(d)

    return data_list, freq


def finalize_vocab_and_tensorize(data_list: List[Data]) -> Tuple[List[Data], Dict[str, int]]:
    """
    Map node_label_str -> node_type ids.
    """
    # build vocab
    labels = set()
    for d in data_list:
        for lb in d.node_label_str:
            labels.add(lb)
    vocab = {lb: i for i, lb in enumerate(sorted(labels))}
    node_vocab_size = len(vocab)

    # edge vocab: PoC uses 1
    edge_vocab_size = 1

    for d in data_list:
        node_type = torch.tensor([vocab[lb] for lb in d.node_label_str], dtype=torch.long)
        d.node_type = node_type
        delattr(d, "node_label_str")

        attach_vocab_sizes(d, node_vocab_size, edge_vocab_size)
        validate_data(d)

    return data_list, vocab


def split_train_test(data_list: List[Data], test_ratio: float, seed: int) -> Tuple[List[Data], List[Data]]:
    n = len(data_list)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_test = int(round(n * test_ratio))
    test_idx = set(perm[:n_test])
    train, test = [], []
    for i, d in enumerate(data_list):
        (test if i in test_idx else train).append(d)
    return train, test


def main():
    ap = argparse.ArgumentParser("Build processed dataset (PoC AST + Bandit)")
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=-1)
    args = ap.parse_args()

    seed_everything(args.seed)

    fps = collect_py_files(args.raw_dir)
    if not fps:
        raise ValueError(f"No .py files found in {args.raw_dir}")

    data_list, _ = build_processed_graphs(fps, max_files=args.max_files)
    data_list, vocab = finalize_vocab_and_tensorize(data_list)

    train_list, test_list = split_train_test(data_list, test_ratio=args.test_ratio, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    torch_save(train_list, os.path.join(args.out_dir, "graphs_train.pt"))
    torch_save(test_list, os.path.join(args.out_dir, "graphs_test.pt"))
    torch_save({"node_vocab": vocab}, os.path.join(args.out_dir, "vocab.pt"))

    print("Done.")
    print("Train:", len(train_list), "Test:", len(test_list))
    print("Node vocab size:", len(vocab), "Edge vocab size:", 1)


if __name__ == "__main__":
    main()

