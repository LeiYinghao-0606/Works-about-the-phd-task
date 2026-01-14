# -*- coding: utf-8 -*-
"""
We implement a robust PoC "guided sampling" for discrete diffusion:
  - Best-of-K sampling (sample-and-rerank)
  - Uses trained guidance classifier (Step B) to score candidates
  - Adds change penalty to discourage unnecessary edits
  - Works with inpainting via data.edit_mask (Step A5)

Assumptions:
  - Graph topology (edge_index) fixed
  - We only edit node_type / edge_type (categorical)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.diffusion.schedule import NoiseSchedule
from src.diffusion.discrete_diffusion import DiscreteInpaintDiffusion
from src.models.denoiser_graph_transformer import GraphTransformerDenoiser, DenoiserConfig
from src.models.guidance_classifier import GuidanceClassifier


# -------------------------
# Checkpoint loading helpers
# -------------------------

def load_denoiser_ckpt(ckpt_path: str, device: str) -> Tuple[GraphTransformerDenoiser, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt or "model_cfg" not in ckpt:
        raise ValueError("Denoiser ckpt missing `model_state` or `model_cfg`.")

    model_cfg = ckpt["model_cfg"]
    cfg = DenoiserConfig(**model_cfg)
    model = GraphTransformerDenoiser(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def load_classifier_ckpt(ckpt_path: str, device: str) -> Tuple[GuidanceClassifier, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("Classifier ckpt missing `model_state`.")

    node_vocab_size = int(ckpt["node_vocab_size"])
    edge_vocab_size = int(ckpt["edge_vocab_size"])
    cfg = ckpt.get("config", {})

    model = GuidanceClassifier(
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("num_layers", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def build_diffusion_from_denoiser_ckpt(den_ckpt: Dict[str, Any], device: str) -> DiscreteInpaintDiffusion:
    """
    Build schedule + diffusion consistent with training config saved in denoiser ckpt.
    """
    cfg = den_ckpt.get("cfg", {})
    T = int(cfg.get("T", 1000))
    beta_start = float(cfg.get("beta_start", 1e-4))
    beta_end = float(cfg.get("beta_end", 2e-2))

    node_vocab_size = int(den_ckpt["node_vocab_size"])
    edge_vocab_size = int(den_ckpt["edge_vocab_size"])

    sched = NoiseSchedule(T=T, beta_start=beta_start, beta_end=beta_end, device=device)
    return DiscreteInpaintDiffusion(
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        schedule=sched,
    )


# -------------------------
# Guidance scoring / rerank
# -------------------------

@dataclass
class GuidanceWeights:
    w_risk: float = 1.0     # multiply predicted risk_score_log (lower is better)
    w_unsafe: float = 0.8   # multiply (1 - safe_prob) (lower is better)
    w_change: float = 0.05  # penalty on edit magnitude


@dataclass
class CandidateScore:
    total: float
    risk_pred: float
    safe_prob: float
    change_ratio: float


@torch.no_grad()
def score_graph(classifier: Optional[GuidanceClassifier], data: Data) -> Tuple[float, float]:
    """
    Returns:
      risk_pred: predicted risk_score_log (float)
      safe_prob: sigmoid(safe_logit) (float)
    If classifier is None, returns (0, 0).
    """
    if classifier is None:
        return 0.0, 0.0
    risk_pred, safe_logit = classifier(data)
    risk_pred = float(risk_pred.view(-1)[0].item())
    safe_prob = float(torch.sigmoid(safe_logit.view(-1)[0]).item())
    return risk_pred, safe_prob


def _default_edge_edit_mask(data: Data, node_edit_mask: torch.Tensor) -> torch.Tensor:
    ei = data.edge_index
    return node_edit_mask[ei[0]] | node_edit_mask[ei[1]]


def change_ratio(x0: Data, x1: Data) -> float:
    """
    Ratio of changed tokens within editable region (nodes + edges averaged).
    """
    node_edit = x0.edit_mask.bool()
    node_den = max(1, int(node_edit.sum().item()))
    node_ratio = float(((x0.node_type != x1.node_type) & node_edit).sum().item()) / float(node_den)

    if hasattr(x0, "edge_edit_mask"):
        edge_edit = x0.edge_edit_mask.bool()
    else:
        edge_edit = _default_edge_edit_mask(x0, node_edit)

    edge_den = max(1, int(edge_edit.sum().item()))
    edge_ratio = float(((x0.edge_type != x1.edge_type) & edge_edit).sum().item()) / float(edge_den)

    return 0.5 * (node_ratio + edge_ratio)


@torch.no_grad()
def best_of_k_inpaint(
    diffusion: DiscreteInpaintDiffusion,
    denoiser: GraphTransformerDenoiser,
    x0: Data,
    *,
    classifier: Optional[GuidanceClassifier] = None,
    K: int = 8,
    steps: int = 200,
    temperature: float = 1.0,
    weights: GuidanceWeights = GuidanceWeights(),
    return_topk: int = 1,
) -> Tuple[Data, List[Tuple[Data, CandidateScore]]]:
    """
    Generate K candidates with diffusion.sample_inpaint() and rerank.

    total_score = w_risk * risk_pred + w_unsafe * (1-safe_prob) + w_change * change_ratio

    Returns:
      best_data (with guidance_* fields attached)
      topk list [(data, score), ...]
    """
    assert K >= 1 and return_topk >= 1
    device = x0.node_type.device

    ranked: List[Tuple[Data, CandidateScore]] = []

    for _ in range(K):
        cand = diffusion.sample_inpaint(
            denoiser,
            x0,
            steps=steps,
            temperature=temperature,
        )
        cr = change_ratio(x0, cand)
        rp, sp = score_graph(classifier, cand)

        total = weights.w_risk * rp + weights.w_unsafe * (1.0 - sp) + weights.w_change * cr
        ranked.append((cand, CandidateScore(float(total), float(rp), float(sp), float(cr))))

    ranked.sort(key=lambda x: x[1].total)

    best = ranked[0][0].clone()
    best.guidance_score = torch.tensor([ranked[0][1].total], dtype=torch.float32, device=device)
    best.guidance_risk_pred = torch.tensor([ranked[0][1].risk_pred], dtype=torch.float32, device=device)
    best.guidance_safe_prob = torch.tensor([ranked[0][1].safe_prob], dtype=torch.float32, device=device)
    best.guidance_change_ratio = torch.tensor([ranked[0][1].change_ratio], dtype=torch.float32, device=device)

    topk = [(d.clone(), s) for (d, s) in ranked[:return_topk]]
    return best, topk

