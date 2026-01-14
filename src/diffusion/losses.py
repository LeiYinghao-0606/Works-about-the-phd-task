# -*- coding: utf-8 -*-
"""
scripts/train_classifier.py

Train guidance classifier on processed graphs:
  - regression: risk_score_log
  - classification: safe_label

Expected fields in Data:
  - node_type, edge_index, edge_type, edit_mask, lineno, node_vocab_size, edge_vocab_size
  - risk_score_log: [1] float
  - safe_label: [1] long
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.loader import DataLoader
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.common.seed import seed_everything
from src.common.io import torch_load, torch_save
from src.common.logger import get_logger
from src.datasets.schema import validate_data
from src.models.guidance_classifier import GuidanceClassifier  # must exist


@dataclass
class Cfg:
    train_path: str = "data/processed/graphs_train.pt"
    out_path: str = "outputs/classifier/guidance_classifier.pt"
    val_ratio: float = 0.1
    seed: int = 42

    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1

    w_reg: float = 1.0
    w_cls: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 50


def split_train_val(data_list: List, val_ratio: float, seed: int):
    n = len(data_list)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(round(n * val_ratio))
    val_idx = set(perm[:n_val])
    tr, va = [], []
    for i, d in enumerate(data_list):
        (va if i in val_idx else tr).append(d)
    return tr, va


@torch.no_grad()
def evaluate(model, loader, device: str) -> Dict[str, float]:
    model.eval()
    reg_sum, cls_sum, n = 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        risk_pred, safe_logit = model(batch)
        y_reg = batch.risk_score_log.view(-1).float()
        y_cls = batch.safe_label.view(-1).float()

        reg_loss = F.mse_loss(risk_pred.view(-1), y_reg)
        cls_loss = F.binary_cross_entropy_with_logits(safe_logit.view(-1), y_cls)

        reg_sum += float(reg_loss.item())
        cls_sum += float(cls_loss.item())
        n += 1
    if n == 0:
        return {"reg": float("inf"), "cls": float("inf")}
    return {"reg": reg_sum / n, "cls": cls_sum / n}


def main():
    ap = argparse.ArgumentParser("Train guidance classifier")
    ap.add_argument("--train_path", type=str, default="data/processed/graphs_train.pt")
    ap.add_argument("--out_path", type=str, default="outputs/classifier/guidance_classifier.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    cfg = Cfg(train_path=args.train_path, out_path=args.out_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)

    seed_everything(cfg.seed)
    logger = get_logger("train_classifier", log_file=os.path.join(os.path.dirname(cfg.out_path), "train.log"))

    data_list = torch_load(cfg.train_path)
    if len(data_list) < 2:
        raise ValueError("Dataset too small.")

    validate_data(data_list[0])

    train_list, val_list = split_train_val(data_list, cfg.val_ratio, cfg.seed)

    node_vocab_size = int(getattr(data_list[0], "node_vocab_size"))
    edge_vocab_size = int(getattr(data_list[0], "edge_vocab_size"))

    model = GuidanceClassifier(
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(train_list, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=cfg.batch_size, shuffle=False)

    best = float("inf")

    # init eval
    ev = evaluate(model, val_loader, cfg.device)
    logger.info(f"[Init] val_reg={ev['reg']:.4f} val_cls={ev['cls']:.4f}")

    step = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            batch = batch.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            risk_pred, safe_logit = model(batch)
            y_reg = batch.risk_score_log.view(-1).float()
            y_cls = batch.safe_label.view(-1).float()

            reg_loss = F.mse_loss(risk_pred.view(-1), y_reg)
            cls_loss = F.binary_cross_entropy_with_logits(safe_logit.view(-1), y_cls)
            loss = cfg.w_reg * reg_loss + cfg.w_cls * cls_loss

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                logger.info(f"[Ep {ep:03d} step {step:06d}] loss={loss.item():.4f} reg={reg_loss.item():.4f} cls={cls_loss.item():.4f}")

        ev = evaluate(model, val_loader, cfg.device)
        score = ev["reg"] + ev["cls"]
        logger.info(f"[Ep {ep:03d} END] val_reg={ev['reg']:.4f} val_cls={ev['cls']:.4f} (best={best:.4f})")

        if score < best:
            best = score
            payload = {
                "model_state": model.state_dict(),
                "node_vocab_size": node_vocab_size,
                "edge_vocab_size": edge_vocab_size,
                "config": {
                    "hidden_dim": cfg.hidden_dim,
                    "num_layers": cfg.num_layers,
                    "dropout": cfg.dropout,
                },
                "best": best,
            }
            torch_save(payload, cfg.out_path)
            logger.info(f"  -> Saved BEST to {cfg.out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

