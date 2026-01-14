# -*- coding: utf-8 -*-
"""
  - Load processed graphs (List[Data])
  - Load denoiser + (optional) classifier
  - For each graph: best-of-K inpaint sampling + rerank
  - Save guided graphs to outputs/guided/graphs_guided.pt
"""

from __future__ import annotations

import os
import argparse
from typing import List, Dict, Any

import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.datasets.loader import load_pyg_list
from src.diffusion.sampler import (
    load_denoiser_ckpt,
    load_classifier_ckpt,
    build_diffusion_from_denoiser_ckpt,
    best_of_k_inpaint,
    GuidanceWeights,
)


def main():
    ap = argparse.ArgumentParser("PromSec-style eval loop (D1: guided graph sampling)")
    ap.add_argument("--in_pt", type=str, default="data/processed/graphs_test.pt")
    ap.add_argument("--out_pt", type=str, default="outputs/guided/graphs_guided.pt")

    ap.add_argument("--denoiser_ckpt", type=str, required=True)
    ap.add_argument("--classifier_ckpt", type=str, default=None)

    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--return_topk", type=int, default=1)
    ap.add_argument("--max_items", type=int, default=-1)

    ap.add_argument("--w_risk", type=float, default=1.0)
    ap.add_argument("--w_unsafe", type=float, default=0.8)
    ap.add_argument("--w_change", type=float, default=0.05)

    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    device = args.device

    # load models + diffusion
    denoiser, den_ckpt = load_denoiser_ckpt(args.denoiser_ckpt, device=device)
    diffusion = build_diffusion_from_denoiser_ckpt(den_ckpt, device=device)

    classifier = None
    if args.classifier_ckpt:
        classifier, _ = load_classifier_ckpt(args.classifier_ckpt, device=device)

    weights = GuidanceWeights(w_risk=args.w_risk, w_unsafe=args.w_unsafe, w_change=args.w_change)

    data_list = load_pyg_list(args.in_pt)
    if args.max_items and args.max_items > 0:
        data_list = data_list[: args.max_items]

    out_list: List[Data] = []
    topk_meta: List[Dict[str, Any]] = []

    for i, d in enumerate(data_list):
        d = d.to(device)

        best, topk = best_of_k_inpaint(
            diffusion,
            denoiser,
            d,
            classifier=classifier,
            K=args.K,
            steps=args.steps,
            temperature=args.temperature,
            weights=weights,
            return_topk=args.return_topk,
        )

        out_list.append(best.to("cpu"))

        if args.return_topk > 1:
            topk_meta.append(
                {
                    "idx": i,
                    "scores": [
                        {
                            "total": s.total,
                            "risk_pred": s.risk_pred,
                            "safe_prob": s.safe_prob,
                            "change_ratio": s.change_ratio,
                        }
                        for (_, s) in topk
                    ],
                }
            )

        if (i + 1) % 10 == 0:
            print(
                f"[{i+1}/{len(data_list)}] "
                f"score={float(out_list[-1].guidance_score.item()):.4f} "
                f"risk={float(out_list[-1].guidance_risk_pred.item()):.4f} "
                f"safe={float(out_list[-1].guidance_safe_prob.item()):.3f} "
                f"chg={float(out_list[-1].guidance_change_ratio.item()):.3f}"
            )

    payload = {"data_list": out_list, "meta": {"topk_scores": topk_meta}}
    torch.save(payload, args.out_pt)
    print("Saved:", args.out_pt, "num_items:", len(out_list))


if __name__ == "__main__":
    main()

