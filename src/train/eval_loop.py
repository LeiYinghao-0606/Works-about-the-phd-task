# -*- coding: utf-8 -*-
"""
PromSec-style iterative loop:
  graph -> (diffusion edit) -> graph' -> (reconstruct code) -> (static analysis) -> select -> repeat

This module is meant to be imported by scripts or notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.diffusion.sampler import best_of_k_inpaint, GuidanceWeights
from src.reconstruction.graph2code_stub import graph2code_stub
from src.analysis.bandit_runner import run_bandit_on_code
from src.analysis.score import bandit_score


@dataclass
class LoopConfig:
    rounds: int = 3
    K: int = 16
    steps: int = 200
    temperature: float = 1.0
    bandit_timeout: int = 30
    guidance_weights: GuidanceWeights = GuidanceWeights()
    context_window: int = 2
    max_lines: int = 80


def rebuild_graph_from_code(code_str: str, x_prev: Data) -> Data:
    """
    Hook for Step A re-build graph from new code.
    Replace this with your CFG/CPG builder and schema alignment.

    Minimal behavior (default):
      - keep previous graph unchanged (no rebuild)
    """
    # For true PromSec behavior, you should:
    #   - parse code_str -> new graph structure
    #   - run analyzer -> new edit_mask/cond_tags
    #   - carry file_path/source_code if needed
    return x_prev


@torch.no_grad()
def run_promsec_loop(
    x0: Data,
    orig_code: str,
    *,
    diffusion,
    denoiser,
    classifier=None,
    cfg: LoopConfig = LoopConfig(),
) -> Tuple[Data, str, Dict[str, Any]]:
    """
    Returns:
      best_graph, best_code, meta
    """
    meta: Dict[str, Any] = {"rounds": []}

    cur_graph = x0.clone()
    cur_code = orig_code

    best_graph = None
    best_code = None
    best_risk = float("inf")

    for r in range(cfg.rounds):
        # D1: sample candidates (top-k list for external scoring)
        _, cand_list = best_of_k_inpaint(
            diffusion,
            denoiser,
            cur_graph,
            classifier=classifier,
            K=cfg.K,
            steps=cfg.steps,
            temperature=cfg.temperature,
            weights=cfg.guidance_weights,
            return_topk=cfg.K,
        )

        round_rows: List[Dict[str, Any]] = []

        # D2 + D3: evaluate each candidate by external verifier
        best_idx = 0
        best_round_risk = float("inf")
        best_round_graph = None
        best_round_code = None
        best_round_vs = None

        for j, (g_cand, gscore) in enumerate(cand_list):
            patched_code, rep = graph2code_stub(
                cur_code,
                x0_graph=cur_graph,
                x1_graph=g_cand,
                context_window=cfg.context_window,
                max_lines=cfg.max_lines,
            )
            br = run_bandit_on_code(patched_code, timeout_sec=cfg.bandit_timeout)
            vs = bandit_score(br)

            row = {
                "cand_rank_guidance": j,
                "guidance_total": gscore.total,
                "guidance_change_ratio": gscore.change_ratio,
                "bandit_issue_count": vs.issue_count,
                "bandit_risk_score": vs.risk_score,
                "bandit_tags": vs.cond_tags[:10],
                "patch_edits": len(rep.edits),
                "bandit_ok": br.ok,
                "bandit_errors": br.errors[:1],
            }
            round_rows.append(row)

            # selection by external risk, tie-break by smaller edits then guidance
            key = (vs.risk_score, row["guidance_change_ratio"], row["guidance_total"])
            if key < (best_round_risk, float("inf"), float("inf")):
                best_round_risk = vs.risk_score
                best_idx = j
                best_round_graph = g_cand.clone()
                best_round_code = patched_code
                best_round_vs = vs

        assert best_round_graph is not None and best_round_code is not None and best_round_vs is not None

        meta["rounds"].append(
            {
                "round": r,
                "best_idx": best_idx,
                "best_risk": best_round_vs.risk_score,
                "best_issue_count": best_round_vs.issue_count,
                "best_tags": best_round_vs.cond_tags[:15],
                "candidates": round_rows,
            }
        )

        # update global best
        if best_round_vs.risk_score < best_risk:
            best_risk = best_round_vs.risk_score
            best_graph = best_round_graph.clone()
            best_code = best_round_code

        # optional: rebuild graph from new code (true PromSec iteration)
        cur_code = best_round_code
        cur_graph = rebuild_graph_from_code(cur_code, best_round_graph)

    assert best_graph is not None and best_code is not None
    return best_graph, best_code, meta

