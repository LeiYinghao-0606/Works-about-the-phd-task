# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from torch_geometric.loader import DataLoader
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.datasets.loader import load_pyg_list, train_val_split
from src.models.guidance_classifier import GuidanceClassifier


@dataclass
class TrainConfig:
    train_path: str = "data/processed/graphs_train.pt"
    out_dir: str = "outputs/classifier"
    val_ratio: float = 0.1
    seed: int = 42

    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-2

    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1

    # multitask loss weights
    alpha_risk: float = 1.0  # MSE
    beta_safe: float = 0.5   # BCE

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _get_targets(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    risk_target: [B] float (risk_score_log)
    safe_target: [B] float (0/1)
    """
    if hasattr(batch, "risk_score_log"):
        risk = batch.risk_score_log.view(-1).float()
    elif hasattr(batch, "risk_score"):
        risk = torch.log1p(batch.risk_score.view(-1).float())
    else:
        raise ValueError("Batch missing risk_score_log or risk_score.")

    if hasattr(batch, "safe_label"):
        safe = batch.safe_label.view(-1).float()
    else:
        # fallback: safe if risk==0
        safe = (risk <= 0.0).float()

    return risk, safe


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    mse = 0.0
    mae = 0.0
    n = 0

    correct = 0
    total = 0

    bce = nn.BCEWithLogitsLoss(reduction="sum")

    bce_sum = 0.0

    for batch in loader:
        batch = batch.to(device)
        risk_t, safe_t = _get_targets(batch)

        risk_p, safe_logit = model(batch)

        # regression metrics
        diff = (risk_p - risk_t)
        mse += (diff * diff).sum().item()
        mae += diff.abs().sum().item()
        n += risk_t.numel()

        # classification metrics
        bce_sum += bce(safe_logit, safe_t).item()
        pred = (torch.sigmoid(safe_logit) >= 0.5).float()
        correct += (pred == safe_t).sum().item()
        total += safe_t.numel()

    return {
        "mse": mse / max(1, n),
        "mae": mae / max(1, n),
        "bce": bce_sum / max(1, total),
        "acc": correct / max(1, total),
    }


def train_classifier(cfg: TrainConfig) -> str:
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    data_list = load_pyg_list(cfg.train_path)
    train_list, val_list = train_val_split(data_list, val_ratio=cfg.val_ratio, seed=cfg.seed)

    # infer vocab sizes from first item (A4 stored these)
    sample = train_list[0]
    node_vocab_size = int(getattr(sample, "node_vocab_size", 0))
    edge_vocab_size = int(getattr(sample, "edge_vocab_size", 0))
    if node_vocab_size <= 0 or edge_vocab_size <= 0:
        raise ValueError("Missing node_vocab_size / edge_vocab_size. Ensure A4 stored them in Data.")

    train_loader = DataLoader(train_list, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=cfg.batch_size, shuffle=False)

    device = cfg.device
    model = GuidanceClassifier(
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, "guidance_classifier.pt")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            risk_t, safe_t = _get_targets(batch)

            risk_p, safe_logit = model(batch)

            loss_r = mse_loss(risk_p, risk_t)
            loss_s = bce_loss(safe_logit, safe_t)
            loss = cfg.alpha_risk * loss_r + cfg.beta_safe * loss_s

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_batches += 1

        val_metrics = evaluate(model, val_loader, device=device)
        train_loss = total_loss / max(1, total_batches)

        # choose best by regression mse (you can also combine)
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_vocab_size": node_vocab_size,
                    "edge_vocab_size": edge_vocab_size,
                    "config": cfg.__dict__,
                    "best_val_mse": best_val,
                },
                best_path,
            )

        print(
            f"[Ep {ep:03d}] train_loss={train_loss:.4f} "
            f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_bce={val_metrics['bce']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"(best_mse={best_val:.4f})"
        )

    return best_path


if __name__ == "__main__":
    cfg = TrainConfig()
    path = train_classifier(cfg)
    print("Saved best model to:", path)

