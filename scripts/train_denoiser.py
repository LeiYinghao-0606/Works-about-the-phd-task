# -*- coding: utf-8 -*-
"""
Train discrete inpaint diffusion denoiser.

Expected Data fields:
  node_type, edge_index, edge_type, edit_mask, lineno, node_vocab_size, edge_vocab_size
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.optim import AdamW

try:
    from torch_geometric.loader import DataLoader
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.common.seed import seed_everything
from src.common.io import torch_load, torch_save
from src.common.logger import get_logger
from src.datasets.schema import validate_data
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.discrete_diffusion import DiscreteInpaintDiffusion
from src.models.denoiser_graph_transformer import GraphTransformerDenoiser, DenoiserConfig


@dataclass
class Cfg:
    train_path: str = "data/processed/graphs_train.pt"
    out_dir: str = "outputs/denoiser"
    val_ratio: float = 0.1
    seed: int = 42

    batch_size: int = 16
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    hidden_dim: int = 192
    num_layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    use_edit_mask_embedding: bool = True

    lambda_node: float = 1.0
    lambda_edge: float = 1.0

    amp: bool = True
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
def evaluate(model, diffusion, loader, cfg: Cfg) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    node_sum = 0.0
    edge_sum = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(cfg.device)
        out = diffusion.training_losses(model, batch, lambda_node=cfg.lambda_node, lambda_edge=cfg.lambda_edge)
        loss_sum += float(out["loss"].item())
        node_sum += float(out["node_loss"].item())
        edge_sum += float(out["edge_loss"].item())
        n += 1
    if n == 0:
        return {"loss": float("inf"), "node": float("inf"), "edge": float("inf")}
    return {"loss": loss_sum / n, "node": node_sum / n, "edge": edge_sum / n}


def main():
    ap = argparse.ArgumentParser("Train denoiser")
    ap.add_argument("--train_path", type=str, default="data/processed/graphs_train.pt")
    ap.add_argument("--out_dir", type=str, default="outputs/denoiser")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    cfg = Cfg(train_path=args.train_path, out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, amp=(not args.no_amp))

    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    logger = get_logger("train_denoiser", log_file=os.path.join(cfg.out_dir, "train.log"))

    data_list = torch_load(cfg.train_path)
    if len(data_list) < 2:
        raise ValueError("Dataset too small.")
    validate_data(data_list[0])

    train_list, val_list = split_train_val(data_list, cfg.val_ratio, cfg.seed)
    train_loader = DataLoader(train_list, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=cfg.batch_size, shuffle=False)

    node_vocab_size = int(getattr(data_list[0], "node_vocab_size"))
    edge_vocab_size = int(getattr(data_list[0], "edge_vocab_size"))

    sched = NoiseSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=cfg.device)
    diffusion = DiscreteInpaintDiffusion(node_vocab_size=node_vocab_size, edge_vocab_size=edge_vocab_size, schedule=sched)

    model_cfg = DenoiserConfig(
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
        use_edit_mask_embedding=cfg.use_edit_mask_embedding,
    )
    model = GraphTransformerDenoiser(model_cfg).to(cfg.device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    best = float("inf")
    best_path = os.path.join(cfg.out_dir, "denoiser_best.pt")
    last_path = os.path.join(cfg.out_dir, "denoiser_last.pt")

    ev0 = evaluate(model, diffusion, val_loader, cfg)
    logger.info(f"[Init] val_loss={ev0['loss']:.4f} node={ev0['node']:.4f} edge={ev0['edge']:.4f}")

    step = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            batch = batch.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = diffusion.training_losses(model, batch, lambda_node=cfg.lambda_node, lambda_edge=cfg.lambda_edge)
                loss = out["loss"]

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                logger.info(
                    f"[Ep {ep:03d} step {step:06d}] loss={loss.item():.4f} "
                    f"node={out['node_loss'].item():.4f} edge={out['edge_loss'].item():.4f} t_mean={out['t_mean'].item():.1f}"
                )

        ev = evaluate(model, diffusion, val_loader, cfg)
        logger.info(f"[Ep {ep:03d} END] val_loss={ev['loss']:.4f} node={ev['node']:.4f} edge={ev['edge']:.4f} (best={best:.4f})")

        # save last
        torch_save(
            {
                "epoch": ep,
                "model_state": model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "cfg": cfg.__dict__,
                "node_vocab_size": node_vocab_size,
                "edge_vocab_size": edge_vocab_size,
                "val": ev,
                "best": best,
            },
            last_path,
        )

        if ev["loss"] < best:
            best = ev["loss"]
            torch_save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "cfg": cfg.__dict__,
                    "node_vocab_size": node_vocab_size,
                    "edge_vocab_size": edge_vocab_size,
                    "val": ev,
                    "best": best,
                },
                best_path,
            )
            logger.info(f"  -> Saved BEST to {best_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

