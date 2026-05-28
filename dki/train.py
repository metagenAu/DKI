"""Training loop and CLI for DKI Phase-1."""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from .data import DKIData, load_dataset
from .device import auto_device
from .infer import predict
from .losses import bray_curtis
from .model import ReplicatorODEFunc


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "results"
    epochs: int = 200
    batch_size: int = 20
    lr: float = 1e-2
    min_lr: float = 1e-4
    weight_decay: float = 0.0
    t_final: float = 100.0
    grad_clip: float = 1.0
    early_stop_patience: int = 50
    val_fraction: float = 0.2
    seed: int = 0
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    device: Optional[str] = None  # auto if None
    save_predictions: bool = True


@dataclass
class TrainResult:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_val_loss: float = math.inf
    best_epoch: int = -1
    epoch_seconds: List[float] = field(default_factory=list)
    final_train_loss: float = math.nan
    final_val_loss: float = math.nan


def _sample_batch(z: torch.Tensor, p: torch.Tensor, batch_size: int, generator: torch.Generator):
    idx = torch.randperm(z.shape[0], generator=generator, device=z.device)[:batch_size]
    return z[idx], p[idx]


def train(cfg: TrainConfig, data: Optional[DKIData] = None) -> tuple[nn.Module, TrainResult, DKIData]:
    device = torch.device(cfg.device) if cfg.device else auto_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if data is None:
        data = load_dataset(cfg.data_dir, val_fraction=cfg.val_fraction, seed=cfg.seed)
    data = data.to(device)
    N = data.n_species

    model = ReplicatorODEFunc(N).to(device)
    optim_ = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    result = TrainResult()
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        z_b, p_b = _sample_batch(data.z_train, data.p_train, cfg.batch_size, gen)

        from .model import integrate

        p_pred = integrate(
            model, z_b, t_final=cfg.t_final, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol
        )
        train_loss = bray_curtis(p_pred, p_b)

        optim_.zero_grad(set_to_none=True)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim_.step()
        scheduler.step()

        # Validation: one batched integration over the whole val set.
        model.eval()
        with torch.no_grad():
            p_val_pred = integrate(
                model,
                data.z_val,
                t_final=cfg.t_final,
                method=cfg.method,
                rtol=cfg.rtol,
                atol=cfg.atol,
            )
            val_loss = bray_curtis(p_val_pred, data.p_val).item()

        result.train_loss.append(train_loss.item())
        result.val_loss.append(val_loss)
        result.epoch_seconds.append(time.perf_counter() - t0)

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(
                f"epoch {epoch:4d}  train_bc={train_loss.item():.4f}  "
                f"val_bc={val_loss:.4f}  best={result.best_val_loss:.4f}  "
                f"sec={result.epoch_seconds[-1]:.2f}"
            )

        if epochs_since_improvement >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.early_stop_patience}).")
            break

    model.load_state_dict(best_state)
    result.final_train_loss = result.train_loss[-1]
    result.final_val_loss = result.best_val_loss

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg.__dict__, "n_species": N},
               out_dir / "best_model.pt")

    if cfg.save_predictions:
        if data.p_test is not None and data.z_test is not None:
            qtst = predict(model, data.z_test, t_final=cfg.t_final, method=cfg.method,
                           rtol=cfg.rtol, atol=cfg.atol)
            np.savetxt(out_dir / "qtst.csv", qtst.detach().cpu().numpy(), delimiter=",")
        qtrn = predict(model, data.z_all, t_final=cfg.t_final, method=cfg.method,
                       rtol=cfg.rtol, atol=cfg.atol)
        np.savetxt(out_dir / "qtrn.csv", qtrn.detach().cpu().numpy(), delimiter=",")

    return model, result, data


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a DKI model.")
    p.add_argument("--data", default="data", help="Directory holding Ptrain.csv etc.")
    p.add_argument("--out", default="results", help="Where to write predictions/checkpoints.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--t-final", type=float, default=100.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="dopri5")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-7)
    p.add_argument("--device", default=None)
    p.add_argument("--no-save-predictions", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        t_final=args.t_final,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        val_fraction=args.val_fraction,
        seed=args.seed,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        device=args.device,
        save_predictions=not args.no_save_predictions,
    )
    _, result, _ = train(cfg)
    print(f"Best val BC: {result.best_val_loss:.6f} at epoch {result.best_epoch}")
    print(f"Mean epoch wall-clock: {np.mean(result.epoch_seconds):.3f}s "
          f"(median {np.median(result.epoch_seconds):.3f}s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
