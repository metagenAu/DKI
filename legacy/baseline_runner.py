"""Adapted copy of the original DKI training loop for head-to-head comparison.

Behavioural changes vs. legacy/DKI_original.py:
  * Reads CSVs directly from ``--data`` (defaults to ``data/``) instead of
    ``../data/<dataset>/``.
  * ``--epochs`` is configurable.
  * Returns timing + final val-loss so we can compare against the new dki/
    package on identical splits and seeds.

Everything else (per-sample odeint, fixed-step Euler-implicit grid t=[0,100]
with dt=0.01 -> 10,000 steps, 2-Linear ODEFunc, BC loss, deepcopy-best-val)
is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint


def loss_bc(p_i, q_i):
    return torch.sum(torch.abs(p_i - q_i)) / torch.sum(torch.abs(p_i + q_i))


def process_data(P):
    Z = P.copy()
    Z[Z > 0] = 1
    P = P / P.sum(axis=0)[np.newaxis, :]
    Z = Z / Z.sum(axis=0)[np.newaxis, :]
    P = P.astype(np.float32)
    Z = Z.astype(np.float32)
    return torch.from_numpy(P.T), torch.from_numpy(Z.T)


class ODEFunc(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.fcc1 = torch.nn.Linear(N, N)
        self.fcc2 = torch.nn.Linear(N, N)

    def forward(self, t, y):  # noqa: ARG002
        out = self.fcc1(y)
        out = self.fcc2(out)
        f = torch.matmul(
            torch.matmul(torch.ones(y.size(dim=1), 1, device=y.device), y),
            torch.transpose(out, 0, 1),
        )
        return torch.mul(y, out - torch.transpose(f, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--out", default="results/legacy")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    P = np.loadtxt(Path(args.data) / "Ptrain.csv", delimiter=",")
    n_cols = P.shape[1]
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(n_cols, size=int(args.val_fraction * n_cols), replace=False)
    train_idx = np.setdiff1d(np.arange(n_cols), val_idx)
    ptrn, ztrn = process_data(P[:, train_idx])
    pval, zval = process_data(P[:, val_idx])
    M, N = ptrn.shape
    ptrn, ztrn = ptrn.to(device), ztrn.to(device)
    pval, zval = pval.to(device), zval.to(device)

    batch_time = 100
    t = torch.arange(0.0, batch_time, 0.01, device=device)

    func = ODEFunc(N).to(device)
    optim_ = torch.optim.Adam(func.parameters(), lr=args.lr)

    loss_train_hist = []
    loss_val_hist = []
    epoch_times = []
    Loss_opt = float("inf")
    best_model = copy.deepcopy(func)

    for e in range(args.epochs):
        t0 = time.perf_counter()
        s = torch.from_numpy(
            np.random.choice(np.arange(M, dtype=np.int64), args.batch_size, replace=False)
        )
        batch_p = ztrn[s, :]
        batch_q = ptrn[s, :]
        batch_t = t

        optim_.zero_grad()
        loss = None
        for i in range(args.batch_size):
            p_pred = odeint(func, batch_p[i].unsqueeze(dim=0), batch_t)
            p_pred = torch.reshape(p_pred[-1, :, :], (1, N))
            term = loss_bc(p_pred.unsqueeze(dim=0), batch_q[i].unsqueeze(dim=0))
            loss = term if loss is None else loss + term

        # Validation
        with torch.no_grad():
            l_val = None
            for i in range(zval.size(dim=0)):
                p_pred = odeint(func, zval[i].unsqueeze(dim=0), batch_t)
                p_pred = torch.reshape(p_pred[-1, :, :], (1, N))
                term = loss_bc(p_pred.unsqueeze(dim=0), pval[i].unsqueeze(dim=0))
                l_val = term if l_val is None else l_val + term
            val_mean = l_val.item() / zval.size(dim=0)

        loss_train_hist.append(loss.item() / args.batch_size)
        loss_val_hist.append(val_mean)
        if val_mean <= Loss_opt:
            Loss_opt = val_mean
            best_model = copy.deepcopy(func)

        loss.backward()
        optim_.step()

        epoch_times.append(time.perf_counter() - t0)
        if e % 5 == 0 or e == args.epochs - 1:
            print(
                f"[legacy] epoch {e:4d}  train_bc={loss_train_hist[-1]:.4f}  "
                f"val_bc={val_mean:.4f}  best={Loss_opt:.4f}  sec={epoch_times[-1]:.2f}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "epoch_times.npy", np.array(epoch_times))
    np.save(out / "val_loss.npy", np.array(loss_val_hist))

    print(f"[legacy] best val BC: {Loss_opt:.6f}")
    print(
        f"[legacy] mean epoch wall-clock: {np.mean(epoch_times):.3f}s "
        f"(median {np.median(epoch_times):.3f}s)"
    )


if __name__ == "__main__":
    main()
