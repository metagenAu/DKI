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
from .infer import equilibrium, predict
from .losses import bray_curtis, composite_loss, self_consistency_loss
from .model import ReplicatorODEFunc


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "results"
    epochs: int = 1000           # one minibatch step per epoch; matches
                                 # the original DKI.py max_epochs and gives
                                 # ~50 passes over a 400-sample train set
                                 # at batch_size=20.
    batch_size: int = 20
    lr: float = 1e-2
    min_lr: float = 1e-4
    weight_decay: float = 0.0
    t_final: float = 100.0
    grad_clip: float = 1.0
    early_stop_patience: int = 200   # val BC is noisy; need a long plateau
                                     # before stopping. Set to a large
                                     # number (or > epochs) to disable.
    val_fraction: float = 0.2
    seed: int = 0
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    device: Optional[str] = None  # auto if None
    save_predictions: bool = True
    # Phase 2: nonlinear fitness + composite loss.
    nonlinear: bool = False
    hidden_mult: int = 2
    loss: str = "bc"             # "bc" or "composite"
    alpha: float = 0.3           # BC weight in the composite loss.
    # Phase 3: deep-equilibrium solver.
    mode: str = "ode"            # "ode" or "deq"
    deq_step: float = 0.5
    deq_max_iter: int = 50
    deq_tol: float = 1e-6
    # Phase 6: self-consistency regulariser (0 disables it).
    consistency_weight: float = 0.0


@dataclass
class TrainResult:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_val_loss: float = math.inf
    best_epoch: int = -1
    epoch_seconds: List[float] = field(default_factory=list)
    final_train_loss: float = math.nan
    final_val_loss: float = math.nan
    skipped_steps: int = 0


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

    model = ReplicatorODEFunc(N, nonlinear=cfg.nonlinear, hidden_mult=cfg.hidden_mult).to(device)
    optim_ = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    def solve(func: nn.Module, z: torch.Tensor) -> torch.Tensor:
        return equilibrium(
            func, z, mode=cfg.mode, t_final=cfg.t_final, method=cfg.method,
            rtol=cfg.rtol, atol=cfg.atol, deq_step=cfg.deq_step,
            deq_max_iter=cfg.deq_max_iter, deq_tol=cfg.deq_tol,
        )

    def supervised_loss(p_pred: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
        if cfg.loss == "composite":
            return composite_loss(p_pred, p_true, alpha=cfg.alpha)
        return bray_curtis(p_pred, p_true)

    result = TrainResult()
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        z_b, p_b = _sample_batch(data.z_train, data.p_train, cfg.batch_size, gen)

        p_pred = solve(model, z_b)
        train_loss = supervised_loss(p_pred, p_b)
        if cfg.consistency_weight > 0:
            train_loss = train_loss + cfg.consistency_weight * self_consistency_loss(
                model, z_b, solve, generator=gen
            )

        optim_.zero_grad(set_to_none=True)
        train_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # A stiff replicator field (common with the nonlinear SiLU fitness) can
        # make a single batch's ODE solve return a non-finite state, hence a
        # non-finite loss/gradient. Stepping on that poisons every weight with
        # NaN via Adam and the whole run reads NaN forever. Skip the update so
        # the optimiser stays on the last good state and the next batch resumes.
        if torch.isfinite(train_loss) and torch.isfinite(grad_norm):
            optim_.step()
        else:
            result.skipped_steps += 1
        scheduler.step()

        # Validation: one batched solve over the whole val set, scored on BC
        # so val numbers stay comparable across loss/mode choices.
        model.eval()
        with torch.no_grad():
            p_val_pred = solve(model, data.z_val)
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
                f"epoch {epoch:4d}  train_loss={train_loss.item():.4f}  "
                f"val_bc={val_loss:.4f}  best={result.best_val_loss:.4f}  "
                f"sec={result.epoch_seconds[-1]:.2f}"
            )

        if epochs_since_improvement >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.early_stop_patience}).")
            break

    if result.skipped_steps:
        print(
            f"Skipped {result.skipped_steps} optimiser step(s) on non-finite "
            "loss/gradient (kept the last good weights). Consider a smaller lr, "
            "lower t_final, or mode='deq' if this happens often."
        )

    model.load_state_dict(best_state)
    result.final_train_loss = result.train_loss[-1]
    result.final_val_loss = result.best_val_loss

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg.__dict__, "n_species": N},
               out_dir / "best_model.pt")

    if cfg.save_predictions:
        pred_kwargs = dict(
            t_final=cfg.t_final, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol,
            mode=cfg.mode, deq_step=cfg.deq_step, deq_max_iter=cfg.deq_max_iter,
            deq_tol=cfg.deq_tol,
        )
        if data.p_test is not None and data.z_test is not None:
            qtst = predict(model, data.z_test, **pred_kwargs)
            np.savetxt(out_dir / "qtst.csv", qtst.detach().cpu().numpy(), delimiter=",")
        qtrn = predict(model, data.z_all, **pred_kwargs)
        np.savetxt(out_dir / "qtrn.csv", qtrn.detach().cpu().numpy(), delimiter=",")

    return model, result, data


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a DKI model.")
    p.add_argument("--data", default="data", help="Directory holding Ptrain.csv etc.")
    p.add_argument("--out", default="results", help="Where to write predictions/checkpoints.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--t-final", type=float, default=100.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="dopri5")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-7)
    p.add_argument("--device", default=None)
    p.add_argument("--no-save-predictions", action="store_true")
    # Phase 2
    p.add_argument("--nonlinear", action="store_true",
                   help="Use fc2(SiLU(fc1(y))) fitness with hidden dim hidden_mult*N.")
    p.add_argument("--hidden-mult", type=int, default=2)
    p.add_argument("--loss", choices=["bc", "composite"], default="bc")
    p.add_argument("--alpha", type=float, default=0.3, help="BC weight in composite loss.")
    # Phase 3
    p.add_argument("--mode", choices=["ode", "deq"], default="ode")
    p.add_argument("--deq-step", type=float, default=0.5)
    p.add_argument("--deq-max-iter", type=int, default=50)
    p.add_argument("--deq-tol", type=float, default=1e-6)
    # Phase 6
    p.add_argument("--consistency-weight", type=float, default=0.0,
                   help="Weight of the self-consistency regulariser (0 disables it).")
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
        nonlinear=args.nonlinear,
        hidden_mult=args.hidden_mult,
        loss=args.loss,
        alpha=args.alpha,
        mode=args.mode,
        deq_step=args.deq_step,
        deq_max_iter=args.deq_max_iter,
        deq_tol=args.deq_tol,
        consistency_weight=args.consistency_weight,
    )
    _, result, _ = train(cfg)
    print(f"Best val BC: {result.best_val_loss:.6f} at epoch {result.best_epoch}")
    print(f"Mean epoch wall-clock: {np.mean(result.epoch_seconds):.3f}s "
          f"(median {np.median(result.epoch_seconds):.3f}s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
