# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from torch_geometric.loader import DataLoader
except Exception as e:
    raise ImportError("torch_geometric is required. Install via: pip install torch-geometric") from e

from src.datasets.loader import load_pyg_list, train_val_split
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.discrete_diffusion import DiscreteInpaintDiffusion
from src.models.denoiser_graph_transformer import GraphTransformerDenoiser, DenoiserConfig


@dataclass
class TrainDenoiserConfig:
    # data
    train_path: str = "data/processed/graphs_train.pt"
    out_dir: str = "outputs/denoiser"
    val_ratio: float = 0.1
    seed: int = 42

    # optimization
    batch_size: int = 16
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    # diffusion schedule
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # denoiser model
    hidden_dim: int = 192
    num_layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    use_edit_mask_embedding: bool = True

    # loss weights
    lambda_node: float = 1.0
    lambda_edge: float = 1.0

    # runtime
    num_workers: int = 0
    pin_memory: bool = True
    amp: bool = True
    log_every: int = 50

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _check_required_fields(d) -> None:
    req = ["node_type", "edge_index", "edge_type", "edit_mask"]
    for k in req:
        if not hasattr(d, k):
            raise ValueError(f"Each Data must contain `{k}`. Missing in an example item.")
    if not hasattr(d, "node_vocab_size") or not hasattr(d, "edge_vocab_size"):
        raise ValueError("Each Data must contain `node_vocab_size` and `edge_vocab_size` (set in Step A4).")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    diffusion: DiscreteInpaintDiffusion,
    loader: DataLoader,
    cfg: TrainDenoiserConfig,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    node_sum = 0.0
    edge_sum = 0.0
    n_batches = 0

    device = cfg.device

    for batch in loader:
        batch = batch.to(device)
        out = diffusion.training_losses(
            model,
            batch,
            lambda_node=cfg.lambda_node,
            lambda_edge=cfg.lambda_edge,
        )
        loss_sum += float(out["loss"].item())
        node_sum += float(out["node_loss"].item())
        edge_sum += float(out["edge_loss"].item())
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("inf"), "node_loss": float("inf"), "edge_loss": float("inf")}

    return {
        "loss": loss_sum / n_batches,
        "node_loss": node_sum / n_batches,
        "edge_loss": edge_sum / n_batches,
    }


def save_ckpt(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def train_denoiser(cfg: TrainDenoiserConfig) -> str:
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "denoiser_best.pt")
    last_path = os.path.join(cfg.out_dir, "denoiser_last.pt")

    # ---- load data ----
    data_list = load_pyg_list(cfg.train_path)
    if len(data_list) < 2:
        raise ValueError(f"Dataset too small: {len(data_list)} items in {cfg.train_path}")

    _check_required_fields(data_list[0])

    train_list, val_list = train_val_split(data_list, val_ratio=cfg.val_ratio, seed=cfg.seed)

    node_vocab_size = int(getattr(data_list[0], "node_vocab_size"))
    edge_vocab_size = int(getattr(data_list[0], "edge_vocab_size"))

    train_loader = DataLoader(
        train_list,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_list,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # ---- build schedule + diffusion ----
    sched = NoiseSchedule(
        T=cfg.T,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=cfg.device,
    )
    diffusion = DiscreteInpaintDiffusion(
        node_vocab_size=node_vocab_size,
        edge_vocab_size=edge_vocab_size,
        schedule=sched,
    )

    # ---- build model ----
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

    best_val = float("inf")
    global_step = 0

    # ---- initial eval ----
    val0 = evaluate(model, diffusion, val_loader, cfg)
    print(f"[Init] val_loss={val0['loss']:.4f} node={val0['node_loss']:.4f} edge={val0['edge_loss']:.4f}")

    # ---- training loop ----
    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        running_node = 0.0
        running_edge = 0.0
        n_log = 0

        for batch in train_loader:
            global_step += 1
            batch = batch.to(cfg.device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                out = diffusion.training_losses(
                    model,
                    batch,
                    lambda_node=cfg.lambda_node,
                    lambda_edge=cfg.lambda_edge,
                )
                loss = out["loss"]

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            running_node += float(out["node_loss"].item())
            running_edge += float(out["edge_loss"].item())
            n_log += 1

            if cfg.log_every > 0 and (global_step % cfg.log_every == 0):
                print(
                    f"[Ep {ep:03d} | step {global_step:06d}] "
                    f"train_loss={running/n_log:.4f} "
                    f"node={running_node/n_log:.4f} edge={running_edge/n_log:.4f} "
                    f"t_mean={float(out['t_mean'].item()):.1f}"
                )
                running = running_node = running_edge = 0.0
                n_log = 0

        # ---- epoch end eval ----
        val = evaluate(model, diffusion, val_loader, cfg)
        print(
            f"[Ep {ep:03d} END] val_loss={val['loss']:.4f} "
            f"node={val['node_loss']:.4f} edge={val['edge_loss']:.4f} "
            f"(best={best_val:.4f})"
        )

        # ---- save last ----
        save_ckpt(
            last_path,
            {
                "epoch": ep,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "cfg": cfg.__dict__,
                "model_cfg": model_cfg.__dict__,
                "node_vocab_size": node_vocab_size,
                "edge_vocab_size": edge_vocab_size,
                "best_val": best_val,
                "val": val,
            },
        )

        # ---- save best ----
        if val["loss"] < best_val:
            best_val = val["loss"]
            save_ckpt(
                best_path,
                {
                    "epoch": ep,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "model_cfg": model_cfg.__dict__,
                    "node_vocab_size": node_vocab_size,
                    "edge_vocab_size": edge_vocab_size,
                    "best_val": best_val,
                    "val": val,
                },
            )
            print(f"  -> Saved BEST to {best_path} (best_val={best_val:.4f})")

    return best_path


def parse_args() -> TrainDenoiserConfig:
    p = argparse.ArgumentParser("Train discrete inpaint diffusion denoiser (CFG tokens)")
    p.add_argument("--train_path", type=str, default="data/processed/graphs_train.pt")
    p.add_argument("--out_dir", type=str, default="outputs/denoiser")
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_edit_mask_embedding", action="store_true")

    p.add_argument("--lambda_node", type=float, default=1.0)
    p.add_argument("--lambda_edge", type=float, default=1.0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = p.parse_args()

    cfg = TrainDenoiserConfig(
        train_path=args.train_path,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        use_edit_mask_embedding=args.use_edit_mask_embedding,
        lambda_node=args.lambda_node,
        lambda_edge=args.lambda_edge,
        amp=args.amp,
        log_every=args.log_every,
        device=args.device,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    best = train_denoiser(cfg)
    print("Training done. Best checkpoint:", best)

